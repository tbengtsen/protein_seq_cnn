#===============================================================
# Tone Bengtsen
# Creation Date: 27-11-2019
#===============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
#  CONV W STRIDE no FC,  seq compressed to 63,
#  channels first increased to 125 then decreased to 8
# =============================================================================
      
class ConvNet(nn.Module):
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
