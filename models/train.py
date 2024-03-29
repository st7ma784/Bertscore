

from pytorch_lightning import LightningModule
import torch
from typing import Optional
from lsafunctions import get_all_LSA_fns
from torch.nn.utils.rnn import pad_sequence
import wandb
class myLightningModule(LightningModule):

    def __init__(self,
                 
                model: Optional[torch.nn.Module] = None, #The bertscore model to use
                LSAVersion: Optional[int] = 0, #The LSA version to use
                all_layers=False,
                perfect_match=False,
                idf_dict={},
                tokenizer=None,
                precision="None",
                **kwargs,
                ):
        super().__init__()
        self.algorithms=get_all_LSA_fns()
        self.algorithms.update({"none":self.no_lsa})
        self.lsa_algorithm=self.algorithms[LSAVersion]
        #self.lsa_algorithm should take a matrix and return a one_hot vector of the same shape.
        self.tokenizer=tokenizer

        self.model=model
        self.all_layers=all_layers
        self.idf_dict=idf_dict
        self.shuffle=perfect_match
        self.precisionfn=self.convert_null
        if precision=="e5m2":
            self.precisionfn=self.convert_to_fp8_e5m2
        elif precision=="e4m3":
            self.precisionfn=self.convert_to_fp8_e4m3


    def convert_to_fp8_e5m2(self,T):
        return T.to(torch.float8_e5m2).to(torch.float32)
    def convert_to_fp8_e4m3(self,T):
        return T.to(torch.float8_e4m3fn).to(torch.float32)
    def convert_null(self,T):
        return T
    def configure_optimizers(self):
        pass 
    def no_lsa(self,tensor):
        return torch.ones_like(tensor)
        
    def bert_encode(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.all_layers)
        if self.all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]
        return emb
    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs)
    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)
        
    def on_test_epoch_start(self, *args, **kwargs):
        self.on_train_epoch_start()
    def on_test_epoch_end(self, *args, **kwargs):
        self.on_train_epoch_end()

    def on_train_epoch_start(self, *args, **kwargs):
        """
        Compute BERTScore.
        Args:
            - :param: `refs` (list of str): reference sentences
            - :param: `hyps` (list of str): candidate sentences
        """
        self.preds = []
        TrueSequenceLength=self.tokenizer.model_max_length if hasattr(self.tokenizer, "model_max_length") else self.tokenizer.max_len
        self.log("SeqLen",TrueSequenceLength)

    def shuffle_batch(self, sen_batch, idf_batch, mask_batch):
        """
        Shuffle the batch for negative sampling.
        Args:
            - :param: `sen_batch` (torch.LongTensor): BxK, batch of sentences
            - :param: `idf_batch` (torch.Tensor): BxK, batch of idf scores
            - :param: `mask_batch` (torch.LongTensor): BxK, batch of masks
        """
        if isinstance(sen_batch, list):
            B=len(sen_batch)
        else:
            B = sen_batch.shape[0] 
        idx = torch.randperm(B)
        sen_batch = torch.stack([sen_batch[i] for i in idx], dim=0)
        idf_batch = torch.stack([idf_batch[i] for i in idx], dim=0)
        mask_batch = torch.stack([mask_batch[i] for i in idx],dim=0)
        return sen_batch, idf_batch, mask_batch
    
    def training_step(self, batch, batch_idx,optimizer_idx=0):

        Hpadded_sens=batch["padded_en"]
        Hpadded_idf=batch["padded_idf_en"]
        Hmasks=batch["mask_en"]
        Rpadded_sens=batch["padded_de"]
        Rpadded_idf=batch["padded_idf_de"]
        Rmasks=batch["mask_de"]
        EOT=self.tokenizer.sep_token_id

        #Get EOT Token id using tokenizer? 
        if isinstance(Hpadded_sens,list):
            #convert to tensor
            Hpadded_sens=torch.stack(Hpadded_sens,dim=1).to(self.device,non_blocking=True)
            Rpadded_sens=torch.stack(Rpadded_sens,dim=1).to(self.device,non_blocking=True)
            Hmasks=torch.stack(Hmasks,dim=1).to(self.device,non_blocking=True)
            Rmasks=torch.stack(Rmasks,dim=1).to(self.device,non_blocking=True)
            Hpadded_idf=torch.stack(Hpadded_idf,dim=1).to(self.device,non_blocking=True)
            Rpadded_idf=torch.stack(Rpadded_idf,dim=1).to(self.device,non_blocking=True)
            
        if self.shuffle:
            Hpadded_sens, Hpadded_idf, Hmasks = self.shuffle_batch(Hpadded_sens, Hpadded_idf, Hmasks)
        #find one hot Hpadded == EOTIDs? 
        REOTLocation=Rpadded_sens==EOT
        HEOTLocation=Hpadded_sens==EOT



        Hembs = self.bert_encode(
            input_ids=Hpadded_sens,
            attention_mask=Hmasks,
        )
        Rembs = self.bert_encode(
            input_ids=Rpadded_sens,
            attention_mask=Rmasks,
        )
        

        P, R, F1,CS= self.greedy_cos_idf(Hembs, Hmasks, Hpadded_idf, Rembs, Rmasks, Rpadded_idf,REOTLocation,HEOTLocation)#pass in EOT IDs        
        P=P.mean()
        R=R.mean()
        F1=F1.mean()
        CS=CS.mean()

        self.preds.append(torch.stack((P, R, F1,CS), dim=-1).cpu())
        #preds = torch.cat(preds, dim=1 if self.all_layers else 0)
        # self.log("P",P, prog_bar=True,enable_graph=False)
        # self.log("R",R, prog_bar=True,enable_graph=False)
        # self.log("F1",F1, prog_bar=True,enable_graph=False)
        # self.log("ClipScore",CS,prog_bar=True,enable_graph=False)
        wandb.log({"P":P,"R":R,"F1":F1,"CS":CS})
        # return {"P":P,"R":R,"F1":F1,"CS":CS}

    def on_train_epoch_end(self, *args, **kwargs):
        """
        Log BERTScore to tensorboard.
        """
        preds = torch.cat(self.preds, dim=0)
        # self.log("e_P",preds[0].mean(),prog_bar=True,enable_graph=False, rank_zero_only=True)
        # self.log("e_R",preds[1].mean(),prog_bar=True,enable_graph=False, rank_zero_only=True)
        # self.log("e_F1",preds[2].mean(),prog_bar=True,enable_graph=False, rank_zero_only=True)
        # self.log("e_ClipScore",preds[3].mean())
        wandb.log({"e_P":preds[0].mean(),"e_R":preds[1].mean(),"e_F1":preds[2].mean(),"e_ClipScore":preds[3].mean()})


    def greedy_cos_idf(
        self,
        ref_embedding,
        ref_masks,
        ref_idf,
        hyp_embedding,
        hyp_masks,
        hyp_idf,
        #recieve EOT IDs
        REOTLocation,
        HEOTLocation,
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
        hyp_embedding=self.precisionfn(hyp_embedding)
        ref_embedding=self.precisionfn(ref_embedding)
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        # masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
        # if self.all_layers:
        #     masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
        # else:
        #     masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

        # masks = masks.float().to(self.device)
        # sim = sim * masks
        # print(sim.shape)
        

        #########Find ClipScore#############
        #HEOTLocation.shape =B,S bool
        # #REOTLocation.shape = B,S bool
        # print(HEOTLocation.shape)
        # print(REOTLocation.shape)
        mask=torch.logical_and(HEOTLocation.unsqueeze(-1),REOTLocation.unsqueeze(1))

        #result is B,S,S
        CS=torch.sum(sim*mask,dim=-1).sum(dim=-1).mean()

        # log CLIP score as EOTid@EOTid? 
        for i in range(sim.shape[0]):
            sim[i]=sim[i]*self.lsa_algorithm(sim[i])
            #there are better ways to do this- the lsa algortihms should all scale to batched just fine. 


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

        return P, R, F,CS


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
    



