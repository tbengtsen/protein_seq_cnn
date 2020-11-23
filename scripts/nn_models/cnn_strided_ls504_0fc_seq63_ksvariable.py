#===============================================================
# Tone Bengtsen
# Creation Date: 27-11-2019
#===============================================================
import numpy as np
import torch
import torch.nn as nn

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
