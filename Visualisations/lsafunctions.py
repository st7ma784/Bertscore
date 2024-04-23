import numpy as np
from typing import Callable

import torch
from functools import reduce
'''
This is research code... it is not clean and it is not commented

If you wish to use it for TPUs, I strongly recommend you refactor your code to use this style of function factory.
 Otherwise your runs will be very slow.

This code is a copy of the LSA methods in the LSA notebook, but with the following changes:
modified to return 1-hot tensors * input so we get a sense of values returned.

'''
from scipy.optimize import linear_sum_assignment

from functools import partial


def outputconversion(func): 
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        output=torch.zeros_like(x)

        x1,y1=func(x, *args, **kwargs)
        try:
            output[x1,y1]=1
        except:
            output[y1,x1]=1
        return output
    return partial(wrapper,func=func)


def to_device(tensor,device):
    return tensor.to(device)


def forcehigh(func):
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        output=func(x, *args, **kwargs)
        output=torch.nonzero(output,as_tuple=True)
        return output
    return partial(wrapper,func=func)

def doFlip(func):
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        out= func(x.T, *args,**kwargs).T if x.shape[0]<x.shape[1] else func(x,*args,**kwargs)
        return out
    return partial(wrapper,func=func)

def get_all_LSA_fns():
    functions={
        "my function": MyLSA,
        "FP8Approximation": FP8LSA,
        "recursive fn":outputconversion(recursiveLinearSumAssignment),
        "recursive fn2 ":outputconversion(recursiveLinearSumAssignment_v2),
        "recursive fn5":outputconversion(recursiveLinearSumAssignment_v5),
        "stock":outputconversion(stock_lsa),

    }
    '''
    "grad_fn":recursiveLinearSumAssignment_grad,
        outputconversion(no_for_loop_MyLinearSumAssignment),
        outputconversion(no_for_loop_triu_MyLinearSumAssignment),
        outputconversion(no_for_loop_v2_MyLinearSumAssignment),
        outputconversion(no_for_loop_v2_triu_MyLinearSumAssignment),
        outputconversion(no_for_loop_v3_MyLinearSumAssignment),
        outputconversion(no_for_loop_v3_triu_MyLinearSumAssignment),
        outputconversion(recursiveLinearSumAssignment_v3),
        outputconversion(recursiveLinearSumAssignment_v4),
    '''
    return functions


def stock_lsa(TruthTensor,maximize=True):
    return linear_sum_assignment(TruthTensor.detach().cpu(),maximize=maximize)


def MyLSA(TruthTensor, maximize=True,lookahead=2):
    '''
    If Maximize is False, I'm trying to minimize the costs. 
    This means that the mask must instead make all the weights far above all the others - 'inf' kind of thing. 
    '''
    mask=torch.zeros(TruthTensor.shape,device=TruthTensor.device,dtype=torch.int8)
    results=torch.zeros_like(TruthTensor)
    finder=torch.argmax if maximize else torch.argmin
    TruthTensor=TruthTensor-torch.min(torch.min(TruthTensor,dim=1,keepdim=True).values,dim=0).values
    replaceval=torch.tensor([float(-1)]) if maximize else torch.max(TruthTensor).to(dtype=torch.float32)+1
    replaceval=replaceval.to(TruthTensor.device)
    dimsizes=torch.tensor(TruthTensor.shape)
    bigdim=torch.argmax(dimsizes)   # 0 
    small_dim=1-bigdim          # 1

    for i in range(TruthTensor.shape[small_dim]): # number of rows 
        array=torch.where(mask==0,TruthTensor,replaceval)
        deltas=torch.diff(torch.topk(array,lookahead,dim=bigdim,largest=maximize).values,n=lookahead-1,dim=bigdim).squeeze()
        col_index=torch.argmax(torch.abs(deltas)) 
        if small_dim==1:
            row_index=finder(array[:,col_index]) 
            results[row_index,col_index]=1
            mask[:,col_index]=1 #mask out the column 
            mask[row_index]=1
        else: 
            row_index=finder(array[col_index])
            results[col_index,row_index]=1

            mask[:,row_index]=1 #mask out the column 
            mask[col_index]=1
    return results


def FP8LSA(TruthTensor, maximize=True):
    '''
    If Maximize is False, I'm trying to minimize the costs. 
    This means that the mask must instead make all the weights far above all the others - 'inf' kind of thing. 
    '''
    mask=torch.zeros(TruthTensor.shape,device=TruthTensor.device,dtype=torch.int8)
    results=torch.zeros_like(TruthTensor)
    finder=torch.argmax if maximize else torch.argmin
    TruthTensor=TruthTensor-torch.min(torch.min(TruthTensor,dim=1,keepdim=True).values,dim=0).values
    replaceval=torch.tensor([float(-1)]) if maximize else torch.max(TruthTensor).to(dtype=torch.float32)+1
    replaceval=replaceval.to(TruthTensor.device)
    dimsizes=torch.tensor(TruthTensor.shape)
    bigdim=torch.argmax(dimsizes)   # 0 
    small_dim=1-bigdim          # 1

    for i in range(TruthTensor.shape[small_dim]): # number of rows 
        array=torch.where(mask==0,TruthTensor,replaceval)
        col_index=finder(finder(array)) 
        if small_dim==1:
            row_index=finder(array[:,col_index]) 
            results[row_index,col_index]=1
            mask[:,col_index]=1 #mask out the column 
            mask[row_index]=1
        else: 
            row_index=finder(array[col_index])
            results[col_index,row_index]=1
            mask[:,row_index]=1 #mask out the column 
            mask[col_index]=1
    return results


def MyLinearSumAssignment(TruthTensor, maximize=True,lookahead=2):
    return MyLSA(TruthTensor, maximize=maximize,lookahead=lookahead).nonzero(as_tuple=True)


def no_for_loop_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.ones_like(rewards,dtype=torch.bool).triu().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values
    Locations=comb_fn(rewards,Costs)
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def no_for_loop_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values 
    Locations=comb_fn(rewards,Costs)
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def no_for_loop_v2_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def no_for_loop_v2_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def no_for_loop_v3_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def no_for_loop_v3_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
   
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def reduceLinearSumAssignment(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove,dim=1):
    removehw,removehwT=remove
    if dim==0:
        removehw,removehwT=removehwT,removehw
    weights=rewards.unsqueeze(0).repeat(*tuple([rewards.shape[0]]+ [1]*len(rewards.shape)))
    weights=weights.masked_fill(removehw,cost_neg)
    Costs=next_highest_fn(weights,dim=dim).values 
    weights2=rewards.T.unsqueeze(0).repeat(*tuple([rewards.shape[1]]+ [1]*len(rewards.shape)))
    weights2=weights2.masked_fill(removehwT,cost_neg)
    Costs2=next_highest_fn(weights2,dim=dim).values
    Cost_total= torch.add(Costs,Costs2.T)
    return Cost_total


def reduceLinearSumAssignment_grad(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove,dim=1):
    removehw,removehwT=remove
    if dim==0:
        removehw,removehwT=removehwT,removehw
    weights=rewards.unsqueeze(0).repeat(*tuple([rewards.shape[0]]+ [1]*len(rewards.shape)))
    weights=weights.masked_fill(removehw,cost_neg)
    Costs=next_highest_fn(weights,dim=dim).values
    weights2=rewards.T.unsqueeze(0).repeat(*tuple([rewards.shape[1]]+ [1]*len(rewards.shape)))
    weights2=weights2.masked_fill(removehwT,cost_neg)
    Costs2=next_highest_fn(weights2,dim=dim).values

    Cost_total= torch.add(Costs,Costs2.T)
    return Cost_total

def reduceLinearSumAssignment_vm(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove:torch.Tensor):
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)
    Costs=next_highest_fn(weights1,dim=1).values  
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total= torch.add(Costs,Costs2)
    return Cost_total


def reduceLinearSumAssignment_v2(rewards:torch.Tensor,maximize=False):
    Topv,topi=rewards.topk(k=2,dim=1,largest=maximize)
    costs=Topv[:,0].unsqueeze(1).repeat(1,rewards.shape[-1])
    one_hot=torch.zeros_like(rewards, dtype=torch.bool).scatter_(1,topi[:,0].unsqueeze(1),1)
    costs[one_hot]=Topv[:,1]
    topv2,topi2=rewards.topk(k=2,dim=0,largest=maximize)
    costs2=topv2[0].unsqueeze(0).repeat(rewards.shape[0],1)
    one_hot2 = torch.zeros_like(rewards, dtype=torch.bool).scatter_(0, topi2[0].unsqueeze(0), 1)
    costs2[one_hot2]=topv2[1]
    Cost_total= costs2+costs
    return Cost_total


def reduceLinearSumAssignment_v3(rewards:torch.Tensor,maximize=True):

    TotalCosts= torch.max(rewards,dim=1,keepdim=True).values + torch.max(rewards,dim=0,keepdim=True).values
    diffs= torch.diff(rewards.topk(k=2,dim=1,largest=maximize).values,dim=1)
    diffs2= torch.diff(rewards.topk(k=2,dim=0,largest=maximize).values,dim=0)
    one_hot=torch.nn.functional.one_hot(torch.argmax(rewards,dim=1),num_classes=rewards.shape[1])
    one_hot=one_hot*diffs
    one_hot2=torch.nn.functional.one_hot(torch.argmax(rewards,dim=0),num_classes=rewards.shape[0])
    one_hot2=one_hot2.T * diffs2
    deltas=one_hot+one_hot2
    totalCosts=TotalCosts+deltas
    return totalCosts

def reduceLinearSumAssignment_vgrad(rewards:torch.Tensor,maximize=True):

    TotalCosts= torch.max(rewards,dim=1,keepdim=True).values + torch.max(rewards,dim=0,keepdim=True).values
    diffs= torch.diff(rewards.topk(k=2,dim=1,largest=maximize).values,dim=1)
    diffs2= torch.diff(rewards.topk(k=2,dim=0,largest=maximize).values,dim=0)
    one_hot=torch.nn.functional.gumbel_softmax(rewards,dim=1,Hard=True)
    one_hot=one_hot*diffs
    one_hot2=torch.nn.functional.gumbel_softmax(rewards,dim=0,Hard=True)
    one_hot2=one_hot2.T * diffs2
    deltas=one_hot+one_hot2
    totalCosts=TotalCosts+deltas
    return totalCosts


def reduceLinearSumAssignment_v4(rewards:torch.Tensor,maximize=True):

    TotalCosts= torch.max(rewards,dim=1,keepdim=True).values + torch.max(rewards,dim=0,keepdim=True).values
    diffs2= torch.diff(rewards.topk(k=2,dim=0,largest=maximize).values,dim=0)
    one_hot2=torch.nn.functional.one_hot(torch.argmax(rewards,dim=0),num_classes=rewards.shape[0])
    one_hot2=one_hot2.T * diffs2
    totalCosts=TotalCosts+one_hot2
    return totalCosts


def recursiveLinearSumAssignment(rewards:torch.Tensor,maximize=True,factor=0.8):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    dimsizes=torch.tensor(rewards.shape)
    bigdim=torch.argmax(dimsizes)
    small_dim=torch.argmin(dimsizes)
    output=torch.zeros_like(rewards,dtype=torch.int8)
    removeHHB=torch.zeros((rewards.shape[small_dim],rewards.shape[small_dim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape) + [rewards.shape[bigdim]]))
    removeBBH=torch.zeros((rewards.shape[bigdim],rewards.shape[bigdim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[small_dim]]))
    for i in range(10):
        cost=reduceLinearSumAssignment(rewards,cost_neg,next_highest_fn,(removeHHB,removeBBH),dim=bigdim)
        rewards=rewards - (cost/factor)
    col_index=final_fn(rewards,dim=bigdim)
    output=(torch.arange(rewards.shape[small_dim],device=rewards.device),col_index) if small_dim==1 else (col_index,torch.arange(rewards.shape[small_dim],device=rewards.device))
    return output


def recursiveLinearSumAssignment_grad(rewards:torch.Tensor,maximize=False,factor=0.8):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    dimsizes=torch.tensor(rewards.shape)
    bigdim=torch.argmax(dimsizes)
    small_dim=torch.argmin(dimsizes)
    output=torch.zeros_like(rewards,dtype=torch.int8)
    removeHHB=torch.zeros((rewards.shape[small_dim],rewards.shape[small_dim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape) + [rewards.shape[bigdim]]))
    removeBBH=torch.zeros((rewards.shape[bigdim],rewards.shape[bigdim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[small_dim]]))
    for i in range(10):
        cost=reduceLinearSumAssignment_vgrad(rewards,cost_neg,next_highest_fn,(removeHHB,removeBBH),dim=bigdim)
        rewards=rewards - (cost/factor)
    output=torch.nn.functional.gumbel_softmax(rewards,dim=bigdim,tau=1, hard=False)
    return output


def recursiveLinearSumAssignment_v2(rewards:torch.Tensor,maximize=True,factor=1):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    dimsizes=torch.tensor(rewards.shape)
    bigdim=torch.argmax(dimsizes)
    small_dim=torch.argmin(dimsizes)
    for i in range(min(rewards.shape[-2:])):
        cost2=reduceLinearSumAssignment_v2(rewards,maximize=maximize)
        rewards=rewards- (cost2/factor)# can remove
    col_index=final_fn(rewards,dim=bigdim)
    output=(torch.arange(rewards.shape[small_dim],device=rewards.device),col_index) if small_dim==1 else (col_index,torch.arange(rewards.shape[small_dim],device=rewards.device))
    return output


def recursiveLinearSumAssignment_v5(rewards:torch.Tensor,maximize=True,factor=10):
    output=torch.zeros_like(rewards,dtype=torch.int8)
    rewards=rewards.clone()
    dimsizes=torch.tensor(rewards.shape)
    small_dim=torch.argmin(dimsizes)
    for i in range(10):
        cost2=reduceLinearSumAssignment_v2(rewards,maximize=maximize)
        rewards=rewards- (cost2/factor)# can remove
    cutoff=torch.topk(rewards.flatten(),rewards.shape[small_dim]+1,largest=maximize,sorted=True).values[-1]
    if maximize:
        output[(rewards>cutoff)]=1
    else:
        output[(rewards<cutoff)]=1
    return output.nonzero(as_tuple=True)


def recursiveLinearSumAssignment_v3(rewards:torch.Tensor,maximize=True,factor=1):
    final_fn=torch.argmax if maximize else torch.argmin
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmax(dimsizes)
    for i in range(rewards.shape[-1]):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
    col_index=final_fn(rewards,dim=dim)        
    return torch.arange(rewards.shape[0],device=rewards.device),col_index

def recursiveLinearSumAssignment_v4(rewards:torch.Tensor,maximize=True,factor=1):
    final_fn=torch.argmax if maximize else torch.argmin
    col_index=None
    dimsizes=torch.tensor(rewards.shape)
    dim=torch.argmin(dimsizes)
    for i in range(rewards.shape[-1]):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
    col_index=final_fn(rewards,dim=dim)
    return torch.arange(rewards.shape[dim],device=rewards.device),col_index

