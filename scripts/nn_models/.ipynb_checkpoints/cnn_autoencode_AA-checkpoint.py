#!/usr/bin/python

#===============================================================
# Tone Bengtsen
# Creation Date: 08-10-2019
#===============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===================================================================
#  Convolutional NN, autoencode channels (AA)
# ===================================================================

class ConvNet(nn.Module):
    def __init__(self,ks_conv,str_conv,pad_conv, ks_pool=None, str_pool=None ,pad_pool=None):
        '''5 layer convolution of channels until 1 chn dimension'''
        super().__init__()
        #self, ks_conv, str_conv, pad_conv, ks_pool, str_pool, pad_pool
        
        self.latent_size = 500 # Must be defined for Tape
        self.layers = 5 # number of layers in encoder
        self.fully_con = 0 # number of fully connected layers in encoder
        self.seq  = 500 # sequence compressed to (encoder)
        self.chnls = 1 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = None # not used, but needed for interface with training
        self.ks_pool = None # not used, but needed for interface with training
        self.pad_pool = None # not used, but needed for interface with training

        
        
        # compress        
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 20, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(20))
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 15, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(15))
        self.conv3 = nn.Sequential(
            nn.Conv1d(15, 10, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(10))
        self.conv4 = nn.Sequential(
            nn.Conv1d(10, 5, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(5)) 
        self.conv5 = nn.Sequential(
            nn.Conv1d(5, 1, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(1)) 
        # decompress 
        self.deconv1 = nn.Sequential(
            nn.Conv1d(1, 5, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(5))  
        self.deconv2 = nn.Sequential(
            nn.Conv1d(5, 10, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(10))
        self.deconv3 = nn.Sequential(
            nn.Conv1d(10, 15, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(15))
        self.deconv4 = nn.Sequential(
            nn.Conv1d(15, 20, self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(20))
        self.deconv5 = nn.Sequential(
            nn.Conv1d(20, 25, self.ks_conv, self.str_conv, padding=self.pad_conv))
    
    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv2(out)  
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

    def decoder(self, out):
        out = self.deconv1(out) 
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        return out

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    def embed(self, x):
        '''embed/encode sequences to latent space. Only added as necessary 
        for TAPE (Nicki's) interface
        '''
        embedding = self.encoder(x)
        # flatten latent space into 2 dim (batch_size*flatten_latent_space)
        batch_size = embedding.shape[0]
        embedding = embedding.view(batch_size,-1)

        return embedding
 
