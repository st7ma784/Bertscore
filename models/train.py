

from pytorch_lightning import LightningModule
import torch
from typing import Optional

class myLightningModule(LightningModule):

    def __init__(self,
                learning_rate,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                context_length= 77,
                **kwargs,
                ):

        super().__init__()
       
    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)

    def training_step(self, batch, batch_idx,optimizer_idx=0):
        
      
    def validation_step(self, batch, batch_idx):
      
       
