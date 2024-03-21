
import torch     
import os
from packaging import version

import pytorch_lightning as pl
from transformers import (
  CLIPTokenizer,
    GPT2Tokenizer,
    RobertaTokenizer
)
from datasets  import load_dataset
from collections import Counter, defaultdict
from itertools import chain
from math import log
from multiprocessing import Pool
from functools import partial
from transformers import __version__ as trans_version
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"]='true'
import json


from torch.utils.data import ConcatDataset, IterableDataset,Dataset

class SummaryDataset(Dataset):
    def __init__(self, HFDataset, idf_dict, tokenizer_fn,tokenizer,seq_len,*args, **kwargs):
        #print('Loading COCO dataset')
        super().__init__(*args, **kwargs)
        self.seq_len=seq_len
        self.tokenizer=tokenizer
        self.tokenize=tokenizer_fn
        self.idf_dict=idf_dict
        self.dataset=HFDataset
        # print(self.dataset.__dir__())
    def sent_encode(self, sent):
        "Encoding as sentence based on the self.tokenizer"
        sent = sent.strip()
        if sent == "":
            print("here!")
            return self.tokenizer.build_inputs_with_special_tokens([])
        return self.tokenize(sent)
    def collate_idf(self,arr):
        """
        Helper function that pads a list of sentences to hvae the same length and
        loads idf score for words in the sentences.

        Args:
            - :param: `arr` (list of str): sentences to process.
            - :param: `numericalize` : a function that takes a list of tokens and
                    return list of token indexes.
        """
        a=arr
        arr_en= self.sent_encode(a["text"])
        arr_de=self.sent_encode(a["summary"])
        idf_weights_en = [self.idf_dict[i] for i in self.sent_encode(a["en"])]
        idf_weights_de = [self.idf_dict[i] for i in self.sent_encode(a["de"])]
        pad_token = self.tokenizer.pad_token_id

        padded_en, lens_en, mask_en = self.padding(arr_en, pad_token, dtype=torch.long)
        padded_idf_en, _, _ = self.padding(idf_weights_en, 0, dtype=torch.float)
        padded_de, lens_de, mask_de = self.padding(arr_de, pad_token, dtype=torch.long)
        padded_idf_de, _, _ = self.padding(idf_weights_de, 0, dtype=torch.float)
        return {
            "padded_en":  padded_en,
            "padded_idf_en":padded_idf_en,
            "mask_en":mask_en,
            "padded_de": padded_de,
            "padded_idf_de":padded_idf_de,
            "mask_de":mask_de,
            
        }
    def padding(self, arr, pad_token, dtype=torch.long):
        # self.lengths.extend(lens)

        slen = min(len(arr),self.seq_len)
        max_len = self.seq_len
        padded = torch.full((max_len,),pad_token, dtype=dtype)
        mask = torch.zeros_like(padded, dtype=torch.long)
        padded[ : slen] = torch.tensor(arr[:slen], dtype=dtype)
        mask=padded!=pad_token
        mask=mask.to(torch.long)
        return padded, torch.tensor(slen), mask

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx: int):
        item=self.dataset.__getitem__(idx)
        return self.collate_idf(item)





class MyDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', batch_size=256,tokenizer=None,**kwargs):
        super().__init__()
        self.data_dir = Cache_dir
        self.batch_size = batch_size
        self.tokenizer =tokenizer
        if self.tokenizer is None: 
            self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        max_len = self.tokenizer.model_max_length if hasattr(self.tokenizer, "model_max_length") else self.tokenizer.max_len
        
        self.seq_len = kwargs.get("padding_length", 15000)

        if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, RobertaTokenizer):
            # for RoBERTa and GPT-2
            if version.parse(trans_version) >= version.parse("4.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, max_length=self.seq_len, truncation=True)

            elif version.parse(trans_version) >= version.parse("3.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, add_prefix_space=True, max_length=self.seq_len, truncation=True,
                    )

            elif version.parse(trans_version) >= version.parse("2.0.0"):
                self.tokenize=partial(self.tokenizer.encode,add_special_tokens=True,add_prefix_space=True,max_length=self.seq_len,)
            else:
                raise NotImplementedError( "transformers version {trans_version} is not supported")
        else:
            if version.parse(trans_version) >= version.parse("4.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, max_length=self.seq_len, truncation=True)
            elif version.parse(trans_version) >= version.parse("3.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, max_length=self.seq_len, truncation=True,)
            elif version.parse(trans_version) >= version.parse("2.0.0"):
                 self.tokenize=partial(self.tokenizer.encode,add_special_tokens=True, max_length=self.seq_len)
            else:
                raise NotImplementedError(
                    f"transformers version {trans_version} is not supported"
                )
        self.lengths=[]
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True,drop_last=True)
    
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
    
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True,drop_last=True)
    
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size
        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=8, prefetch_factor=4, pin_memory=True,drop_last=True)
    
    def sent_encode(self, sent):
        "Encoding as sentence based on the self.tokenizer"
        sent = sent.strip()
        if sent == "":
            print("here!")
            return self.tokenizer.build_inputs_with_special_tokens([])
        return self.tokenize(sent)
    def collate_idf(self,arr):
        """
        Helper function that pads a list of sentences to hvae the same length and
        loads idf score for words in the sentences.

        Args:
            - :param: `arr` (list of str): sentences to process.
            - :param: `numericalize` : a function that takes a list of tokens and
                    return list of token indexes.
        """
        arr = [(self.sent_encode(a["text"]),
                   self.sent_encode(a["summary"]))
                    if ("en" in a and
                     "de" in a) else (None,None) for a in arr["translation"]
        ]
        [arr_en,arr_de] = zip(*arr)
        idf_weights_en = [[self.idf_dict[i] for i in a] for a in arr_en]
        idf_weights_de = [[self.idf_dict[i] for i in a] for a in arr_de]
        pad_token = self.tokenizer.pad_token_id

        padded_en, lens_en, mask_en = self.padding(arr_en, pad_token, dtype=torch.long)
        padded_idf_en, _, _ = self.padding(idf_weights_en, 0, dtype=torch.float)
        padded_de, lens_de, mask_de = self.padding(arr_de, pad_token, dtype=torch.long)
        padded_idf_de, _, _ = self.padding(idf_weights_de, 0, dtype=torch.float)
        return {
            "padded_en":  padded_en,
            "padded_idf_en":padded_idf_en,
            "mask_en":mask_en,
            "padded_de": padded_de,
            "padded_idf_de":padded_idf_de,
            "mask_de":mask_de,
            
        }
    def padding(self, arr, pad_token, dtype=torch.long):
        # self.lengths.extend(lens)

        lens = torch.LongTensor([len(a) for a in arr]).clamp(max=self.seq_len)
        max_len = self.seq_len
        padded = torch.full((len(arr), max_len),pad_token, dtype=dtype)
        mask = torch.zeros_like(padded, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, : lens[i]] = torch.tensor(a[:lens[i]], dtype=dtype)
        
        mask=padded!=pad_token
        mask=mask.to(torch.long)
        return padded, lens, mask
    def prepare_data(self):

        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        #2014 german to english
        self.data = load_dataset("billsum",                                
                               split='train',
                               cache_dir=self.data_dir,
                               streaming=False,)
        #print(self.data.__dir__())
        self.get_idf_dict(self.data['train'])

    def tokenization(self,sample):

        return self.collate_idf(sample)

    def sent_encode(self, sent):
        "Encoding as sentence based on the self.tokenizer"
        sent = sent.strip()
        if sent == "":
            print("here!")
            return self.tokenizer.build_inputs_with_special_tokens([])
        return self.tokenize(sent)
    def batch_encode(self, sents):
        "Encoding as batch based on the self.tokenizer"
        return [self.sent_encode(sent) for sent in sents]


    def process(self,a):
                  
    
        return chain.from_iterable([ self.batch_encode(a[key]) for key in a.keys()])
               
                
    def get_idf_dict(self, arr):
        """
        Returns mapping from word piece index to its inverse document frequency.


        Args:
            - :param: `arr` (list of str) : sentences to process.
            - :param: `self.tokenizer` : a BERT self.tokenizer corresponds to `model`.
            - :param: `nthreads` (int) : number of CPU threads to use
        """
        tokenizername = type(self.tokenizer).__name__
        split_name="Billsum"
        #check for existing idf_dict
        self.idf_dict=defaultdict(lambda: log((len(arr) + 1) / len(arr)))
        if os.path.exists(f"{self.data_dir}/{split_name}_{tokenizername}_idf_dict2.json"):
            with open(f"{self.data_dir}/{split_name}_{tokenizername}_idf_dict2.json", "r") as f:
                idf_dict=json.load(f)
        else:
            idf_count = Counter()
            num_docs = len(arr)
            # cpu_count=os.cpu_count()
            # with Pool(cpu_count) as p:
            #use map instead
            dataloader= torch.utils.data.DataLoader(arr, batch_size=200, shuffle=False, num_workers=4, prefetch_factor=3, pin_memory=True,drop_last=False)
            for a in tqdm(dataloader):
                for tokens in self.process(a):
                    idf_count.update(tokens)
    #       idf_count.update(chain.from_iterable(p.map(self.process, arr)))
            idf_dict = {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
            with open(f"{self.data_dir}/{split_name}_{tokenizername}_idf_dict2.json", "w") as f:
                json.dump(idf_dict,f)
        self.idf_dict.update(idf_dict)
        return 


    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        from datasets import load_dataset

        if not hasattr(self,"data"):
            self.data=load_dataset("billsum",                                
                               split='train',
                               cache_dir=self.data_dir,
                               streaming=True,)
            self.get_idf_dict(self.data['train'])
            self.dataset=self.data.map(lambda x: self.collate_idf(x), batched=True, remove_columns=["text","summary","title"])
        else:
            self.dataset=SummaryDataset(self.data['train'],self.idf_dict,self.tokenize,tokenizer=self.tokenizer,seq_len=self.seq_len)
        train_size = int(0.99 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train,self.test = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        #self.test=test_dataset
    
        '''With retrospect, should have used the split api built into HF. Oh Well!'''



if __name__=="__main__":


    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='location of data')
    parser.add_argument('--data', type=str, default='/data', help='location of data')
    args = parser.parse_args()
    print("args",args)
    datalocation=args.data
    datamodule=MyDataModule(Cache_dir=datalocation,batch_size=2)

    datamodule.prepare_data()
    datamodule.setup()
    dl=datamodule.train_dataloader()
    for batch in dl:
        print(batch)

