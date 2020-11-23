#===============================================================
# Tone Bengtsen
# Creation Date: 27-11-2019
#===============================================================

import argparse
import os 
import numpy as np

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datetime
import utils
from utils import *
from torchsummary import summary
from contextlib import redirect_stdout
from matplotlib import pyplot as plt

# =============================================================================
#  CONV W AVG POOL +1 FC, seq compressed to 63, channels to 100, FC 6300->500
# =============================================================================
class ConvAvgPool_w_FC(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool, str_pool, pad_pool):
        '''5 layer convolutions, 3 of them  with avg pooling, then one fully connnected. 
        Latent Space = 500'''
        super().__init__()
        
        self.latent_size = 500 # MUST BE DEFINED
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
        
        # define layers in model
        ## ENCODER ##
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50)
            )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool , self.str_pool, self.pad_pool ),
            nn.Conv1d(50,50, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50)
            )
        self.conv3 = nn.Sequential(
            nn.Conv1d(50, 100, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100)
            )
        self.conv_pool4 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100) )
        
        self.conv_pool5 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
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
        
        self.deconv_up5 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))

        
        self.deconv_up4 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        
        self.deconv3 = nn.Sequential(
            nn.Conv1d(100, 50, self.ks_conv, self.str_conv , padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
        self.deconv_up2 = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, 3, 1 , 1),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25, 3, 1 , 1))        
   


    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv_pool2(out)
        out = self.conv3(out)
        out = self.conv_pool4(out)
        out = self.conv_pool5(out)
        out = out.view(-1, out.shape[1]*out.shape[2]) # flatten array but not batches
        out = self.fc1_encode(out)
        return out  
        
    def decoder(self, out):
#        out = self.fc2_decode(out)
        out = self.fc1_decode(out)
        out = out.view(-1, 100,63) # re-shape array but not batches
        out = self.deconv_up5(out)
        out = self.deconv_up4(out)
        out = self.deconv3(out)
        out = self.deconv_up2(out)
        out = self.deconv1(out)
        return out

 
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out 
#
    def embed(self, x):
        '''embed/encode sequences to latent space. Only added as necessary 
        for TAPE (Nicki's) interface
        '''
        embedding = self.encoder(x)
        # flatten latent space into 2 dim (batch_size*flatten_latent_space)
        batch_size = embedding.shape[0]
        embedding = embedding.view(batch_size,-1)

        return embedding

# =============================================================================
#       CONV W AVG POOL no FC, seq compressed to 4, channels to 100
# =============================================================================

class ConvAvgPool_no_FC_seq4(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool, str_pool, pad_pool):
        '''9 layer convolutions, 7 of them with avg pooling.
        Compresses sequence to 4, increase channels to 100. =>
        Latent space 4*100=400  '''
        super().__init__()
        
        self.latent_size = 400 # MUST BE DEFINED
        self.layers = 9 # number of layers in encoder
        self.fully_con = 0 # number of fully connected layers in encoder
        self.seq  = 4 # sequence compressed to (encoder)
        self.chnls = 100 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool
                
        # define layers in model
        ## ENCODE ##
        
        # input [batchsize, channels, seq_len]
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50)
            )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool , self.str_pool, self.pad_pool ),
            nn.Conv1d(50,50, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50)
            )
        self.conv3 = nn.Sequential(
            nn.Conv1d(50, 100, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100)
            )
        self.conv_pool4 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        
        self.conv_pool5 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))

        self.conv_pool6 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))   
             
         # [100, 100, 32] -> [100, 100, 16]
        self.conv_pool7 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        # [100, 100, 16] -> [100, 100, 8]
        self.conv_pool8 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))

        # [100, 100, 8] -> [100, 100, 4]
        self.conv_pool9 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(100, 100, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))


       ### DECODE ### 
        self.deconv_up9 = nn.Sequential(
            nn.Upsample(8, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        self.deconv_up8 = nn.Sequential(
            nn.Upsample(16, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))

        self.deconv_up7 = nn.Sequential(
            nn.Upsample(32, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))

        self.deconv_up6 = nn.Sequential(
            nn.Upsample(63, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))        

        self.deconv_up5 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))

        self.deconv_up4 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(100, 100, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        
        self.deconv3 = nn.Sequential(
            nn.Conv1d(100, 50, self.ks_conv, self.str_conv , padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
        self.deconv_up2 = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25, self.ks_pool, 1, padding=self.pad_pool))        
   


    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv_pool2(out)
        out = self.conv_pool4(out)
        out = self.conv_pool5(out)
        out = self.conv_pool6(out)
        out = self.conv_pool7(out)
        out = self.conv_pool8(out)
        out = self.conv_pool9(out)
        return out  
        
    def decoder(self, out):
        out = self.deconv_up9(out)
        out = self.deconv_up8(out)
        out = self.deconv_up7(out)
        out = self.deconv_up6(out)
        out = self.deconv_up5(out)
        out = self.deconv_up4(out)
        out = self.deconv_up2(out)
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
    
    
# =============================================================================
#       CONV W AVG POOL no FC, seq compressed to 8, channels to 50
# =============================================================================


class ConvAvgPool_no_FC_seq8(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool, str_pool, pad_pool):
        '''7 layer convolutions, 6 of them with avg pooling.
        Compresses sequence to 8, increase channels to 50. =>
        Latent space 8*50=400  '''
        super().__init__()
        
        self.latent_size = 400 # MUST BE DEFINED        
        self.layers = 7 # number of layers in encoder
        self.fully_con = 0 # number of fully connected layers in encoder
        self.seq  = 8 # sequence compressed to (encoder)
        self.chnls = 50 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool
        
        # define layers in model
        ## ENCODE ##
        
        # input [batchsize, channels, seq_len]
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50)
            )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool , self.str_pool, self.pad_pool ),
            nn.Conv1d(50,50, self.ks_conv,  self.str_conv, self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50)
            )
#        self.conv3 = nn.Sequential(
#            nn.Conv1d(50, 100, self.ks_conv,  self.str_conv, self.pad_conv),
#            nn.ReLU(),
#            nn.BatchNorm1d(100)
#            )
        self.conv_pool4 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(50, 50, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
        self.conv_pool5 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(50, 50, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))

        self.conv_pool6 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(50, 50, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))   
             
         # [100, 100, 32] -> [100, 100, 16]
        self.conv_pool7 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(50, 50, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        # [100, 50, 16] -> [100, 50, 8]
        self.conv_pool8 = nn.Sequential(
            nn.AvgPool1d(self.ks_pool, self.str_pool, self.pad_pool),
            nn.Conv1d(50, 50, self.ks_conv,  self.str_conv,self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
       ### DECODE ### 
        self.deconv_up8 = nn.Sequential(
            nn.Upsample(16, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(50))

        self.deconv_up7 = nn.Sequential(
            nn.Upsample(32, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(50))

        self.deconv_up6 = nn.Sequential(
            nn.Upsample(63, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(50))        

        self.deconv_up5 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(50))

        self.deconv_up4 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
#        self.deconv3 = nn.Sequential(
#            nn.Conv1d(100, 50, self.ks_conv, self.str_conv , padding=self.pad_conv),
#            nn.ReLU(),
#            nn.BatchNorm1d(50))
        
        self.deconv_up2 = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50, self.ks_pool, 1, padding=self.pad_pool),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25, self.ks_pool, 1, padding=self.pad_pool))        
   


    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv_pool2(out)
        out = self.conv_pool4(out)
        out = self.conv_pool5(out)
        out = self.conv_pool6(out)
        out = self.conv_pool7(out)
        out = self.conv_pool8(out)
        return out  
        
    def decoder(self, out):
        out = self.deconv_up8(out)
        out = self.deconv_up7(out)
        out = self.deconv_up6(out)
        out = self.deconv_up5(out)
        out = self.deconv_up4(out)
        out = self.deconv_up2(out)
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