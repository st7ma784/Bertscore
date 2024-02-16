

from pytorch_lightning import LightningModule
import torch
from typing import Optional
from lsafunctions import get_all_LSA_fns
from torch.nn.utils.rnn import pad_sequence

class myLightningModule(LightningModule):

    def __init__(self,
                 
                model: Optional[torch.nn.Module] = None, #The bertscore model to use
                LSAVersion: Optional[int] = 0, #The LSA version to use
                all_layers=False,
                perfect_match=False,
                idf_dict={},
                **kwargs,
                ):
        super().__init__()
        self.algorithms=get_all_LSA_fns()
        self.algorithms.update({"none":self.no_lsa})
        self.lsa_algorithm=self.algorithms[LSAVersion]
        #self.lsa_algorithm should take a matrix and return a one_hot vector of the same shape.
        self.model=model
        self.all_layers=all_layers
        self.idf_dict=idf_dict
        self.shuffle=perfect_match
        
    def no_lsa(self,tensor):
        return torch.ones_like(tensor)
        
    def bert_encode(self, x, attention_mask):
        with torch.no_grad():
            out = self(x, attention_mask=attention_mask, output_hidden_states=self.all_layers)
        if self.all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]
        return emb
    def forward(self,*args,**kwargs):
        return self.model(input)
    def train_step(self, batch, batch_idx):
        pass

    def on_test_epoch_start(self, *args, **kwargs):
        """
        Compute BERTScore.
        Args:
            - :param: `refs` (list of str): reference sentences
            - :param: `hyps` (list of str): candidate sentences
        """
        self.preds = []

    def shuffle_batch(self, sen_batch, idf_batch, mask_batch):
        """
        Shuffle the batch for negative sampling.
        Args:
            - :param: `sen_batch` (torch.LongTensor): BxK, batch of sentences
            - :param: `idf_batch` (torch.Tensor): BxK, batch of idf scores
            - :param: `mask_batch` (torch.LongTensor): BxK, batch of masks
        """
        B, K = sen_batch.size()
        idx = torch.randperm(B)
        return sen_batch[idx], idf_batch[idx], mask_batch[idx]
    def test_step(self, batch, batch_idx,optimizer_idx=0):

        Hpadded_sens=batch["padded_en"]
        Hpadded_idf=batch["padded_idf_en"]
        Hmasks=batch["mask_en"]
        Rpadded_sens=batch["padded_de"]
        Rpadded_idf=batch["padded_idf_de"]
        Rmasks=batch["mask_de"]
        
        if self.shuffle:
            Hpadded_sens, Hpadded_idf, Hmasks = self.shuffle_batch(Hpadded_sens, Hpadded_idf, Hmasks)



        Hembs = self.bert_encode(
            Hpadded_sens,
            attention_mask=Hmasks,
        )
        Rembs = self.bert_encode(
            Rpadded_sens,
            attention_mask=Rmasks,
        )
        

        P, R, F1 = self.greedy_cos_idf(Hembs, Hmasks, Hpadded_idf, Rembs, Rmasks, Rpadded_idf)        
        preds.append(torch.stack((P, R, F1), dim=-1).cpu())
        preds = torch.cat(preds, dim=1 if self.all_layers else 0)
        self.log("P",P)
        self.log("R",R)
        self.log("F1",F1)
        return preds

    def on_test_epoch_end(self, *args, **kwargs):
        """
        Log BERTScore to tensorboard.
        """
        preds = torch.cat(self.preds, dim=1 if self.all_layers else 0)
        self.log("e_P",preds[0].mean())
        self.log("e_R",preds[1].mean())
        self.log("e_F1",preds[2].mean())



    def greedy_cos_idf(
        self,
        ref_embedding,
        ref_masks,
        ref_idf,
        hyp_embedding,
        hyp_masks,
        hyp_idf,

    ):
        """
        Compute greedy matching based on cosine similarity.

        Args:
            - :param: `ref_embedding` (torch.Tensor):
                    embeddings of reference sentences, BxKxd,
                    B: batch size, K: longest length, d: bert dimenison
            - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                    reference sentences.
            - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                    piece in the reference setence
            - :param: `hyp_embedding` (torch.Tensor):
                    embeddings of candidate sentences, BxKxd,
                    B: batch size, K: longest length, d: bert dimenison
            - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                    candidate sentences.
            - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                    piece in the candidate setence
        """
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        if self.all_layers:
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
        if self.all_layers:
            masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
        else:
            masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

        masks = masks.float().to(self.device)
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
        if self.all_layers:
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

        # hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
        # ref_zero_mask = ref_masks.sum(dim=1).eq(2)

        if self.all_layers:
            P = P.view(L, B)
            R = R.view(L, B)
            F = F.view(L, B)

        F = F.masked_fill(torch.isnan(F), 0.0)

        return P, R, F


    # def pad_batch_stats(self,sen_batch, stats_dict):
    #     stats = [stats_dict[s] for s in sen_batch]
    #     emb, idf = zip(*stats)
    #     emb = [e.to(self.device) for e in emb]
    #     idf = [i.to(self.device) for i in idf]
    #     lens = [e.size(0) for e in emb]
    #     emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
    #     idf_pad = pad_sequence(idf, batch_first=True)

    #     def length_to_mask(lens):
    #         lens = torch.tensor(lens, dtype=torch.long)
    #         max_len = max(lens)
    #         base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
    #         return base < lens.unsqueeze(1)

    #     pad_mask = length_to_mask(lens).to(self.device)
    #     return emb_pad, pad_mask, idf_pad
    



