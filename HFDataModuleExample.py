
import torch     
import os
import pytorch_lightning as pl
from transformers import (
  CLIPTokenizer
)
os.environ["TOKENIZERS_PARALLELISM"]='true'

class MyDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', batch_size=256,ZHtokenizer=None,ENtokenizer=None):
        super().__init__()
        self.data_dir = Cache_dir
        self.batch_size = batch_size
        self.Tokenizer =CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
    
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
        from datasets import load_dataset
        #2014 german to english
        self.dataset = load_dataset("wmt16", "de-en",
                               cache_dir=self.data_dir,
                               streaming=False,
                               )
   
    def tokenization(self,sample):

        return {'en' : self.tokenizer(sample["en"],padding="max_length",
                            truncation=True,
                            max_length=77,
                            return_tensors="pt" 
                            )["input_ids"],
                'de' : self.tokenizer(sample["de"], 
                            padding="max_length",
                            truncation=True,
                            max_length=77,
                            return_tensors="pt" 
                            )["input_ids"]}
        
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        from datasets import load_dataset

        if not hasattr(self,"dataset"):
            self.dataset=load_dataset("wmt16", "de-en",
                                 cache_dir=self.data_dir,
                                 streaming=True,
                                 )
        #MAP ITEM -> [{'en' : item.split("|||")[0], 'zh' : item.split("|||")[1]} for item in self.dataset['train']['translation']]   
        reformatted_dataset = self.dataset["train"].map(lambda x: {'en' : x["text"].split("|||")[0], 'de' : x["text"].split("|||")[1]})
        #remove the old "text" column
        reformatted_dataset.remove_columns("text")
        #tokenize the reformatted dataset
        self.train=reformatted_dataset.map(lambda x: self.tokenization(x), batched=True)
        self.val=reformatted_dataset.map(lambda x: self.tokenization(x), batched=True)
        self.test = reformatted_dataset.map(lambda x: self.tokenization(x), batched=True)
        
        
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
