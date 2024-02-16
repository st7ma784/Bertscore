
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
os.environ["self.tokenizerS_PARALLELISM"]='true'
import json
class MyDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', batch_size=256,tokenizer=None,**kwargs):
        super().__init__()
        self.data_dir = Cache_dir
        self.batch_size = batch_size
        self.tokenizer =tokenizer
        if self.tokenizer is None: 
            self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, RobertaTokenizer):
            # for RoBERTa and GPT-2
            if version.parse(trans_version) >= version.parse("4.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, max_length=self.tokenizer.model_max_length, truncation=True)

            elif version.parse(trans_version) >= version.parse("3.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, add_prefix_space=True, max_length=self.tokenizer.max_len, truncation=True,
                    )

            elif version.parse(trans_version) >= version.parse("2.0.0"):
                self.tokenize=partial(self.tokenizer.encode,add_special_tokens=True,add_prefix_space=True,max_length=self.tokenizer.max_len,)
            else:
                raise NotImplementedError( "transformers version {trans_version} is not supported")
        else:
            if version.parse(trans_version) >= version.parse("4.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, max_length=self.tokenizer.model_max_length, truncation=True)
            elif version.parse(trans_version) >= version.parse("3.0.0"):
                self.tokenize=partial(self.tokenizer.encode, add_special_tokens=True, max_length=self.tokenizer.max_len, truncation=True,)
            elif version.parse(trans_version) >= version.parse("2.0.0"):
                 self.tokenize=partial(self.tokenizer.encode,add_special_tokens=True, max_length=self.tokenizer.max_len)
            else:
                raise NotImplementedError(
                    f"transformers version {trans_version} is not supported"
                )
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True,drop_last=True)
    
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
    
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=False, num_workers=2, prefetch_factor=2, pin_memory=True,drop_last=True)
    
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size
        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    
    def prepare_data(self):

        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        #2014 german to english
        self.dataset = load_dataset("wmt16", "de-en",
                               cache_dir=self.data_dir,
                               streaming=False,
                               )
        self.get_idf_dict(self.dataset['train'])
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
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

    def collate_idf(self,arr):
        """
        Helper function that pads a list of sentences to hvae the same length and
        loads idf score for words in the sentences.

        Args:
            - :param: `arr` (list of str): sentences to process.
            - :param: `numericalize` : a function that takes a list of tokens and
                    return list of token indexes.
        """
        # print(arr) #len is 1?
        # for a in arr:
        #     if isinstance(a,str):
        #         print(a)
        #     if isinstance(a,dict):
        #         if "translation" in a:
        #             print("translation",a["translation"])
                
        arr = [(self.sent_encode(a["en"]),
                   self.sent_encode(a["de"]))
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
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, : lens[i]] = 1
        return padded, lens, mask

    def process(self,a):
                  
    
        return chain.from_iterable([ self.batch_encode(a["translation"][key]) for key in a["translation"].keys()])
               
                
    def get_idf_dict(self, arr):
        """
        Returns mapping from word piece index to its inverse document frequency.


        Args:
            - :param: `arr` (list of str) : sentences to process.
            - :param: `self.tokenizer` : a BERT self.tokenizer corresponds to `model`.
            - :param: `nthreads` (int) : number of CPU threads to use
        """
        tokenizername = type(self.tokenizer).__name__
        split_name="wmt16de-en"
        #check for existing idf_dict
        self.idf_dict=defaultdict(lambda: log((len(arr) + 1) / len(arr)))
        if os.path.exists(f"{self.data_dir}/{split_name}_{tokenizername}_idf_dict2.json"):
            with open(f"{self.data_dir}/{split_name}_{tokenizername}_idf_dict2.json", "r") as f:
                idf_dict=json.load(f)
                self.idf_dict.update(idf_dict)
            return
        idf_count = Counter()
        num_docs = len(arr)
        # cpu_count=os.cpu_count()
        # with Pool(cpu_count) as p:
        #use map instead
        dataloader= torch.utils.data.DataLoader(arr, batch_size=100, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True,drop_last=False)
        for a in tqdm(dataloader):
            for tokens in self.process(a):
                idf_count.update(tokens)
#       idf_count.update(chain.from_iterable(p.map(self.process, arr)))
        idf_dict = {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
        with open(f"{self.data_dir}/{split_name}_{tokenizername}_idf_dict2.json", "w") as f:
            json.dump(idf_dict,f)
        self.idf_dict.update(idf_dict)
    

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        from datasets import load_dataset

        if not hasattr(self,"dataset"):
            self.dataset=load_dataset("wmt16", "de-en",
                               cache_dir=self.data_dir,
                               streaming=True,)
        #get the idf dictionary
            self.get_idf_dict(self.dataset['train'])
        #MAP ITEM -> [{'en' : item.split("|||")[0], 'zh' : item.split("|||")[1]} for item in self.dataset['train']['translation']]   
        # reformatted_dataset = self.dataset["train"].map(lambda x: {'en' : x["text"].split("|||")[0], 'de' : x["text"].split("|||")[1]})
        #remove the old "text" column
        # reformatted_dataset.remove_columns("text")
        #tokenize the reformatted dataset
        self.train=self.dataset['train'].map(lambda x: self.tokenization(x), batched=True)
        self.val=self.train
        self.test=self.train
        
        
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
    for batch in tqdm(dl):
        print(batch)

