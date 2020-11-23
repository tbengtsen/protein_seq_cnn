#===============================================================
# Tone Bengtsen
# Creation Date: 27-11-2019
#===============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
#   CONV W STRIDE + 1xFC,  seq compressed to 63, channels to 100, FC 6300->500
# =============================================================================

class ConvNet(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool=None, str_pool=None, pad_pool=None):
        '''5 layer convolutions, 3 of them  with stride, then 1 fully connnected
        fc: 6300-> 500. 
        Latent Space = 500'''
       
        super().__init__()
        
        self.latent_size = 500 # Must be defined for Tape
        self.layers = 5 # number of layers in encoder
        self.fully_con = 1 # number of fully connected layers in encoder
        self.seq  = 63 # sequence compressed to (encoder)
        self.chnls = 100 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool

        ## ENCODER ##       
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50,   self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv1a = nn.Sequential(
            nn.Conv1d(50, 50,   self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 100,   self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        self.conv3 = nn.Sequential(
            nn.Conv1d(100,100,   self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(100,100,   self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        
        self.fc1_encode = nn.Sequential(
            nn.Linear(100*63, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500)) 
       
        ## DECODER ## 
        self.fc1_decode = nn.Sequential(
            nn.Linear(500, 100*63),
            nn.ReLU(),
            nn.BatchNorm1d(100*63))
        
        self.deconv4 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(100, 100,   self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        self.deconv3 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(100, 100,   self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        self.deconv2 = nn.Sequential(
            nn.Conv1d(100, 50,   self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1a = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50,   self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25,   self.ks_conv, 1, padding=self.pad_conv))
    

    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv1a(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(-1, out.shape[1]*out.shape[2]) 
        out = self.fc1_encode(out)
        return out

    def decoder(self, out):
        out = self.fc1_decode(out)
        out = out.view(-1, 100,63) # re-shape array but not batches
        out = self.deconv4(out)
        out = self.deconv3(out)
        out = self.deconv2(out)
        out = self.deconv1a(out)
        out = self.deconv1(out)

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
