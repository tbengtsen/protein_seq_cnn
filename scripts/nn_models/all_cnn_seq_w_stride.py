#===============================================================
# Tone Bengtsen
# Creation Date: 27-11-2019
#===============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Contains NN model architecturs all with stride. 
# Following models build: 
# - ConvStride_w_FC (5 layer conv, 3 of which with stride, then 1 fully conv)
# - ConvStride_w_2_FC (5 layer conv, 3 of which with stride, then 2 fully conv)
# - ConvStride_no_FC_seq4 (9 layer conv, 7 of which with stride) 
# - ConvStride_no_FC_seq63 (8 layer conv, 3 of which with stride, compress seq 2 63, then channels to 8) 
# - ConvStride_no_FC_seq63_ks_variable (same as above but with decreasing KS)
# =============================================================================


# =============================================================================
#   CONV W STRIDE + 1xFC,  seq compressed to 63, channels to 100, FC 6300->500
# =============================================================================

class ConvStride_w_FC(nn.Module):
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


# =============================================================================
#   CONV W STRIDE + 2xFC,  seq compressed to 63, channels to 100, FC 6300->500
# =============================================================================
class ConvStride_w_2_FC(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool=None, str_pool=None, pad_pool=None):
        '''5 layer convolutions, 3 of them  with stride 2. 
        Then 2 fully connnected.  fc: 6300-> 1000, fc2:1000-> 500
        Latent Space = 500'''
        super().__init__()

        self.latent_size = 500 # MUST BE DEFINED
        self.layers = 5 # number of layers in encoder
        self.fully_con = 2 # number of fully connected layers in encoder
        self.seq  = 63 # sequence compressed to (encoder)
        self.chnls = 100 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool
        

        ## compress ##       
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50,  self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv1a = nn.Sequential(
            nn.Conv1d(50, 50,  self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 100,  self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        self.conv3 = nn.Sequential(
            nn.Conv1d(100,100,  self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(100,100,  self.ks_conv, self.str_conv, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        
        self.fc1_encode = nn.Sequential(
            nn.Linear(100*63, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000)) 
        self.fc2_encode = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500)) 
    
        
        ## decompress ## 
        self.fc2_decode = nn.Sequential(
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000)) 
        self.fc1_decode = nn.Sequential(
            nn.Linear(1000, 100*63),
            nn.ReLU(),
            nn.BatchNorm1d(100*63))
        
        self.deconv4 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(100, 100,  self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        self.deconv3 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(100, 100,  self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(100))
        self.deconv2 = nn.Sequential(
            nn.Conv1d(100, 50,  self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1a = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50,  self.ks_conv, 1, padding=self.pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25,  self.ks_conv, 1, padding=self.pad_conv))
    

    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv1a(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(-1, out.shape[1]*out.shape[2]) 
        out = self.fc1_encode(out)
        out = self.fc2_encode(out)

        return out

    def decoder(self, out):
        out = self.fc2_decode(out)
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


# =============================================================================
#   CONV W STRIDE no FC,  seq compressed to 4, channels to 125. 
# =============================================================================
    
class ConvStride_no_FC_seq4(nn.Module):
    def __init__(self,ks_conv, str_conv, pad_conv, ks_pool=None, str_pool=None, pad_pool=None):
        '''9 layer convolutions, 7 of them with stride 2.
        Compresses sequence to 4, increase channels to 125. =>
        Latent space 4*125=500  '''
        super().__init__()

        self.latent_size = 500 # MUST BE DEFINED
        self.layers = 9 # number of layers in encoder
        self.fully_con = 0 # number of fully connected layers in encoder
        self.seq  = 4 # sequence compressed to (encoder)
        self.chnls = 125 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool

        ## compress ##       
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv1a = nn.Sequential(
            nn.Conv1d(50, 50,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv3 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv4 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv5 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv6 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv7 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv8 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        
        ## DECODE ## 
        self.deconv8 = nn.Sequential(
            nn.Upsample(8, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv7 = nn.Sequential(
            nn.Upsample(16, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv6 = nn.Sequential(
            nn.Upsample(32, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv5 = nn.Sequential(
            nn.Upsample(63, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))

        self.deconv4 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv3 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv2 = nn.Sequential(
            nn.Conv1d(125, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1a = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25,  ks_conv, 1, padding=pad_conv))
    

    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv1a(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        return out

    def decoder(self, out):
        out = self.deconv8(out)
        out = self.deconv7(out)
        out = self.deconv6(out)
        out = self.deconv5(out)
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

# =============================================================================
#  CONV W STRIDE no FC,  seq compressed to 63,
#  channels first increased to 125 then decreased to 8
# =============================================================================
      
class ConvStride_no_FC_seq63(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool=None, str_pool=None, pad_pool=None):
        '''8 layer convolutions, 3 of them with str_conv 2 -> seq compressed to 63.
        Channels first increased to 125, then compressed to 8 =>
        Latent space 63*8=504  '''
        super().__init__()
        
        self.latent_size = 504 # MUST BE DEFINED
        self.layers = 8 # number of layers in encoder
        self.fully_con = 0 # number of fully connected layers in encoder
        self.seq  = 63 # sequence compressed to (encoder)
        self.chnls = 8 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool
        
        ## ENCODER ##       
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv1a = nn.Sequential(
            nn.Conv1d(50, 50,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv3 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv4 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv5 = nn.Sequential(
            nn.Conv1d(125,50,  ks_conv,1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv6 = nn.Sequential(
            nn.Conv1d(50,25 , ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(25))
        self.conv7 = nn.Sequential(
            nn.Conv1d(25,8,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(8))
        
        ## DECODER ## 
        self.deconv7 = nn.Sequential(
            nn.Conv1d(8,25,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(25))
        self.deconv6 = nn.Sequential(
            nn.Conv1d(25,50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv5 = nn.Sequential(
            nn.Conv1d(50,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv4 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv3 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv2 = nn.Sequential(
            nn.Conv1d(125, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1a = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25,  ks_conv, 1, padding=pad_conv))
    

    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv1a(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return out

    def decoder(self, out):
        out = self.deconv7(out)
        out = self.deconv6(out)
        out = self.deconv5(out)
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
        for TAPE (Nicki's) interface. Obs latent space must be flattend
        '''
        embedding = self.encoder(x)
        # flatten latent space into 2 dim (batch_size*flatten_latent_space)
        batch_size = embedding.shape[0]
        embedding = embedding.view(batch_size,-1)

        return embedding

    
    
# =============================================================================
#  CONV W STRIDE no FC,  seq compressed to 63,
#  channels first increased to 125 then decreased to 8
# kernelsize variable, decreases with conv. 
# =============================================================================

class ConvStride_no_FC_seq63_ks_variable(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool=None, str_pool=None, pad_pool=None):
        '''8 layer convolutions, 3 of them with stride 2 -> seq compressed to 63.
        Channels first increased to 125, then compressed to 8 =>
        Kernel size variable and decreasing with conv layers
        Latent space 63*8=504  '''
        
        super().__init__()
        
        self.latent_size = 504 # MUST BE DEFINED
        self.layers = 8 # number of layers in encoder
        self.fully_con = 0 # number of fully connected layers in encoder
        self.seq  = 63 # sequence compressed to (encoder)
        self.chnls = 8 # latent space chnls 
        # define model params 
        self.ks_conv = None
        self.str_conv = 2
        self.pad_conv = None
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool
        
        
        ## ENCODER ##       
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50,  11, 1, padding=5),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv1a = nn.Sequential(
            nn.Conv1d(50, 50,  3, str_conv, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 125, 11 , 1, padding=5),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv3 = nn.Sequential(
            nn.Conv1d(125,125,  3, str_conv, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv4 = nn.Sequential(
            nn.Conv1d(125,125, 5, str_conv, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv5 = nn.Sequential(
            nn.Conv1d(125,50, 3,1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv6 = nn.Sequential(
            nn.Conv1d(50,25 , 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(25))
        self.conv7 = nn.Sequential(
            nn.Conv1d(25,8,  3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(8))
        
        ## DECODER ##
        self.deconv7 = nn.Sequential(
            nn.Conv1d(8,25,  3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(25))
        self.deconv6 = nn.Sequential(
            nn.Conv1d(25,50,  3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv5 = nn.Sequential(
            nn.Conv1d(50,125, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        
        self.deconv4 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  5, 1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv3 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(125,125, 5, 1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv2 = nn.Sequential(
            nn.Conv1d(125, 50,  11, 1, padding=5),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1a = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50,  3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25,  11, 1, padding=5))
    

    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv1a(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return out

    def decoder(self, out):
        out = self.deconv7(out)
        out = self.deconv6(out)
        out = self.deconv5(out)
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


# =============================================================================
#  CONV W STRIDE no FC,  seq compressed to 32,
#  channels first increased to 125 then decreased to 8
# kernelsize variable, decreases with conv. 
# =============================================================================


    
class ConvStride_no_FC_seq32(nn.Module):
    def __init__(self, ks_conv, str_conv, pad_conv, ks_pool=None, str_pool=None, pad_pool=None):
        '''9 layer convolutions, 3 of them with stride 2 -> seq compressed to
        32.
        Channels first increased to 125, then compressed to 8 =>
        Latent space 32*12=512 '''
        super().__init__()
        self.latent_size = 512 # MUST BE DEFINED
        self.layers = 9 # number of layers in encoder
        self.fully_con = 0 # number of fully connected layers in encoder
        self.seq  = 32 # sequence compressed to (encoder)
        self.chnls = 16 # latent space chnls 
        # define model params 
        self.ks_conv = ks_conv
        self.str_conv = str_conv
        self.pad_conv = pad_conv
        self.str_pool = str_pool
        self.ks_pool = ks_pool
        self.pad_pool = pad_pool
        

        ## ENCODER ##       
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 50,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv3 = nn.Sequential(
            nn.Conv1d(50, 125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv4 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv5 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv6 = nn.Sequential(
            nn.Conv1d(125,125,  ks_conv, str_conv, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv7 = nn.Sequential(
            nn.Conv1d(125,50,  ks_conv,1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv8 = nn.Sequential(
            nn.Conv1d(50,25 , ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(25))
        self.conv9 = nn.Sequential(
            nn.Conv1d(25,16,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(16))
        
        ## DECODER ## 
        self.deconv9 = nn.Sequential(
            nn.Conv1d(16,25,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(25))
        self.deconv8 = nn.Sequential(
            nn.Conv1d(25,50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv7 = nn.Sequential(
            nn.Conv1d(50,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv6 = nn.Sequential(
            nn.Upsample(63, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
        self.deconv5 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv4 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv3 = nn.Sequential(
            nn.Conv1d(125, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv2 = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50,  ks_conv, 1, padding=pad_conv),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25,  ks_conv, 1, padding=pad_conv))
    

    def encoder(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        return out

    def decoder(self, out):
        out = self.deconv9(out)
        out = self.deconv8(out)
        out = self.deconv7(out)
        out = self.deconv6(out)
        out = self.deconv5(out)
        out = self.deconv4(out)
        out = self.deconv3(out)
        out = self.deconv2(out)
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
