

from pytorch_lightning import LightningModule
import torch
from typing import Optional
from lsafunctions import get_all_LSA_fns
from torch.nn.utils.rnn import pad_sequence

class myLightningModule(LightningModule):

    def __init__(self,
                model: Optional[torch.nn.Module] = None, #The bertscore model to use
                LSAVersion: Optional[int] = 0, #The LSA version to use
                **kwargs,
                ):
        super().__init__()
        self.algorithms=get_all_LSA_fns()
        self.algorithms.update({0:self.no_lsa})
        self.lsa_algorithm=self.algorithms[LSAVersion]
        #self.lsa_algorithm should take a matrix and return a one_hot vector of the same shape.

    def no_lsa(self,tensor):
        return torch.ones_like(tensor)
        
    def bert_encode(self, x, attention_mask, all_layers=False):
        with torch.no_grad():
            out = self(x, attention_mask=attention_mask, output_hidden_states=all_layers)
        if all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]
        return emb
    def forward(self,input):
        return self.model(input)

    def training_step(self, batch, batch_idx,optimizer_idx=0):
        pass
      
      
        

    def greedy_cos_idf(
        self,
        ref_embedding,
        ref_masks,
        ref_idf,
        hyp_embedding,
        hyp_masks,
        hyp_idf,
        all_layers=False,

    ):
        """
        Compute greedy matching based on cosine similarity.

        Args:
            - :param: `ref_embedding` (torch.Tensor):
                    embeddings of reference sentences, BxKxd,
                    B: batch size, K: longest length, d: bert dimenison
            - :param: `ref_lens` (list of int): list of reference sentence length.
            - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                    reference sentences.
            - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                    piece in the reference setence
            - :param: `hyp_embedding` (torch.Tensor):
                    embeddings of candidate sentences, BxKxd,
                    B: batch size, K: longest length, d: bert dimenison
            - :param: `hyp_lens` (list of int): list of candidate sentence length.
            - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                    candidate sentences.
            - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                    piece in the candidate setence
        """
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        if all_layers:
            B, _, L, D = hyp_embedding.size()
            hyp_embedding = (
                hyp_embedding.transpose(1, 2)
                .transpose(0, 1)
                .contiguous()
                .view(L * B, hyp_embedding.size(1), D)
            )
            ref_embedding = (
                ref_embedding.transpose(1, 2)
                .transpose(0, 1)
                .contiguous()
                .view(L * B, ref_embedding.size(1), D)
            )
        batch_size = ref_embedding.size(0)
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
        if all_layers:
            masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
        else:
            masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

        masks = masks.float().to(sim.device)
        sim = sim * masks
        LSA=self.lsa_algorithm(sim)
        #LSA is a onehot vector of the best match
        sim = sim * LSA


        word_precision = sim.max(dim=2)[0]
        word_recall = sim.max(dim=1)[0]

        hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
        ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
        precision_scale = hyp_idf.to(self.device)
        recall_scale = ref_idf.to(self.device)
        if all_layers:
            precision_scale = (
                precision_scale.unsqueeze(0)
                .expand(L, B, -1)
                .contiguous()
                .view_as(word_precision)
            )
            recall_scale = (
                recall_scale.unsqueeze(0).expand(L, B, -1).contiguous().view_as(word_recall)
            )
        P = (word_precision * precision_scale).sum(dim=1)
        R = (word_recall * recall_scale).sum(dim=1)
        F = 2 * P * R / (P + R)

        hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
        ref_zero_mask = ref_masks.sum(dim=1).eq(2)

        if all_layers:
            P = P.view(L, B)
            R = R.view(L, B)
            F = F.view(L, B)

        if torch.any(hyp_zero_mask):
            print(
                "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.",
                file=sys.stderr,
            )
            P = P.masked_fill(hyp_zero_mask, 0.0)
            R = R.masked_fill(hyp_zero_mask, 0.0)

        if torch.any(ref_zero_mask):
            print(
                "Warning: Empty reference sentence detected; setting raw BERTScores to 0.",
                file=sys.stderr,
            )
            P = P.masked_fill(ref_zero_mask, 0.0)
            R = R.masked_fill(ref_zero_mask, 0.0)

        F = F.masked_fill(torch.isnan(F), 0.0)

        return P, R, F


    def bert_cos_score_idf(
        self,
        refs,
        hyps,
        tokenizer,
        idf_dict,
        batch_size=64,
        all_layers=False,
    ):
        """
        Compute BERTScore.

        Args:
            - :param: `model` : a BERT model in `pytorch_pretrained_bert`
            - :param: `refs` (list of str): reference sentences
            - :param: `hyps` (list of str): candidate sentences
            - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
            - :param: `idf_dict` : a dictionary mapping a word piece index to its
                                inverse document frequency
            - :param: `verbose` (bool): turn on intermediate status update
            - :param: `batch_size` (int): bert score processing batch size
            - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
        """
        preds = []

        def dedup_and_sort(l):
            return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

        sentences = dedup_and_sort(refs + hyps)
        embs = []
        iter_range = range(0, len(sentences), batch_size)
       
        stats_dict = dict()
        for batch_start in iter_range:
            sen_batch = sentences[batch_start : batch_start + batch_size]
            embs, masks, padded_idf = self.get_bert_embedding(sen_batch, idf_dict,all_layers=all_layers)
            embs = embs.cpu()
            masks = masks.cpu()
            padded_idf = padded_idf.cpu()
            for i, sen in enumerate(sen_batch):
                sequence_len = masks[i].sum().item()
                emb = embs[i, :sequence_len]
                idf = padded_idf[i, :sequence_len]
                stats_dict[sen] = (emb, idf)



        iter_range = range(0, len(refs), batch_size)

        with torch.no_grad():
            for batch_start in iter_range:
                batch_refs = refs[batch_start : batch_start + batch_size]
                batch_hyps = hyps[batch_start : batch_start + batch_size]
                ref_stats = self.pad_batch_stats(batch_refs, stats_dict)
                hyp_stats = self.pad_batch_stats(batch_hyps, stats_dict)

                P, R, F1 = self.greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
                preds.append(torch.stack((P, R, F1), dim=-1).cpu())
        preds = torch.cat(preds, dim=1 if all_layers else 0)
        return preds

    def pad_batch_stats(self,sen_batch, stats_dict):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(self.device) for e in emb]
        idf = [i.to(self.device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(self.device)
        return emb_pad, pad_mask, idf_pad
    




    def collate_idf(self,arr, idf_dict):
        """
        Helper function that pads a list of sentences to hvae the same length and
        loads idf score for words in the sentences.

        Args:
            - :param: `arr` (list of str): sentences to process.
            - :param: `tokenize` : a function that takes a string and return list
                    of tokens.
            - :param: `numericalize` : a function that takes a list of tokens and
                    return list of token indexes.
            - :param: `idf_dict` (dict): mapping a word piece index to its
                                inverse document frequency
            - :param: `pad` (str): the padding token.
            - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
        """
        arr = [sent_encode(self.tokenizer, a) for a in arr]

        idf_weights = [[idf_dict[i] for i in a] for a in arr]

        pad_token = self.tokenizer.pad_token_id

        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = self.padding(idf_weights, 0, dtype=torch.float)

        padded = padded.to(self.device)
        mask = mask.to(self.device)
        lens = lens.to(self.device)
        return padded, padded_idf, lens, mask


    def get_bert_embedding(
        self,
        all_sens,
        idf_dict,
        batch_size=-1,
        all_layers=False,
    ):
        """
        Compute BERT embedding in batches.

        Args:
            - :param: `all_sens` (list of str) : sentences to encode.
            - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
            - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
            - :param: `idf_dict` (dict) : mapping a word piece index to its
                                inverse document frequency
            - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
        """

        padded_sens, padded_idf, _, mask = self.collate_idf(all_sens, idf_dict)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = self.bert_encode(
                    self.model,
                    padded_sens[i : i + batch_size],
                    attention_mask=mask[i : i + batch_size],
                    all_layers=all_layers,
                )
                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=0)

        return total_embedding, mask, padded_idf


    def padding(self, arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, : lens[i]] = 1
        return padded, lens, mask


