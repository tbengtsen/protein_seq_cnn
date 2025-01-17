# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:33:27 2019

@author: nsde
"""
import numpy as np
from torch import nn
from .datasets import get_dataset
from scipy.stats import spearmanr

from .utils import accuracy_topk

#%%
class Task(nn.Module):
    def __init__(self, embed_model, fix_embedding=True):
        super().__init__()
        self.embed_model = embed_model
        self.latent_size = self.embed_model.latent_size
        self.fix_embedding = fix_embedding
        if fix_embedding:
            for p in self.embed_model.parameters():
                p.requires_grad = False
        
        
    def forward(self, batch):
        embedding = self.embed_model.embed(batch['primary'])
        return self.predictor(embedding)
        
    def loss_func(self, batch):
        raise NotImplementedError
        
    def get_data(self, batch_size=10, max_length=500):
        raise NotImplementedError
        
#%%
class StabilityTask(Task):
    def __init__(self, embed_model, fix_embedding=True):
        super().__init__(embed_model, fix_embedding)
    
        self.predictor = nn.Sequential(nn.LayerNorm(self.latent_size),
                                       nn.Linear(self.latent_size, 500),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(500, 1))
    
        self.loss = nn.MSELoss()
    
    def loss_func(self, batch):
        prediction =  self(batch)
        target = batch['stability_score']
        loss = self.loss(prediction, target)
        mae = (prediction-target).abs().mean()
        corr, _ = spearmanr(prediction.detach().cpu().numpy(), target.cpu().numpy())
        metrics = {'MSE': loss.item(), 'MAE': mae.item(), 'S_Corr': corr}
        return loss, metrics
    
    def get_data(self, batch_size=10, max_length=500):
        dataset = get_dataset('stability')(batch_size=batch_size)
        return dataset.train_set, dataset.val_set, dataset.test_set
    
    # added by Tone
    def get_prediction(self,batch):
        '''returns  predictions and target data. 
        Used for test performance output'''
        prediction =  self(batch)
        prediction = prediction.detach().cpu().numpy()
        target = batch['stability_score']
        target = target.cpu().numpy()

        return prediction.reshape(-1), target.reshape(-1)

        #return np.concatenate((prediction,target),axis=1)
        
        
#%%
class FluorescenceTask(Task):
    def __init__(self, embed_model, fix_embedding=True):
        super().__init__(embed_model, fix_embedding)
        
        self.predictor = nn.Sequential(nn.LayerNorm(self.latent_size),
                                       nn.Linear(self.latent_size, 500),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(500, 1))
    
        self.loss = nn.MSELoss()
        
    def loss_func(self, batch):
        prediction = self(batch)
        target = batch['log_fluorescence']
        loss = self.loss(prediction, target)
        mae = (prediction-target).abs().mean()
        corr, _ = spearmanr(prediction.detach().cpu().numpy(), target.cpu().numpy())
        metrics = {'MSE': loss.item(), 'MAE': mae.item(), 'S_Corr': corr}
        return loss, metrics
    
    def get_data(self, batch_size=10, max_length=500):
        dataset = get_dataset('fluorescence')(batch_size=batch_size)
        return dataset.train_set, dataset.val_set, dataset.test_set
    
    # added by Tone
    def get_prediction(self,batch):
        '''returns  predictions and target data. 
        Used for test performance output'''
        prediction =  self(batch)
        prediction = prediction.detach().cpu().numpy()
        target = batch['log_fluorescence']
        target = target.cpu().numpy()
        return prediction.reshape(-1), target.reshape(-1)

#%%
class RemotehomologyTask(Task):
    def __init__(self, embed_model, fix_embedding=True):
        super().__init__(embed_model, fix_embedding)
            
        self.predictor = nn.Sequential(nn.LayerNorm(self.latent_size),
                                       nn.Linear(self.latent_size, 500),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(500, 1195))
        
        self.loss = nn.CrossEntropyLoss()
        
    def loss_func(self, batch):
        prediction = self(batch)
        target = batch['fold_label'].long()
        loss = self.loss(prediction, target)
        acc = (prediction.argmax(dim=-1) == target).float().mean()
        top5, top10 = accuracy_topk(prediction, target, topk=(5,10))
        metrics = {'CrossEntro': loss, 'Acc': acc.item(), 
                   'Top5Acc': top5.item(), 'Top10Acc': top10.item()}
        return loss, metrics
    
    def get_data(self, batch_size=10, max_length=500):
        dataset = get_dataset('remotehomology')(batch_size=batch_size)
        return dataset.train_set, dataset.val_set, dataset.test_set
    
    # added by Tone
    def get_prediction(self,batch):
        '''returns  predictions and target data. 
        Used for test performance output'''
        prediction =  self(batch)
        prediction = prediction.detach().cpu().numpy()
        target = batch['fold_label'].long()
        target = target.cpu().numpy()
        return prediction.reshape(-1), target.reshape(-1)


#%%
def get_task(name):
    d = {'fluorescence': FluorescenceTask,
         #'proteinnet': ProteinnetDataset,
         'remotehomology': RemotehomologyTask,
         #'secondarystructure': SecondarystructureDataset,
         'stability': StabilityTask,
         }
    assert name in d, '''Unknown task, choose from {0}'''.format(
            [k for k in d.keys()])
    return d[name]
