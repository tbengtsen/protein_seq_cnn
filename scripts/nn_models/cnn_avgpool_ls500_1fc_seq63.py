#===============================================================
# Tone Bengtsen
# Creation Date: 27-11-2019
#===============================================================
import torch
import torch.nn as nn

# =============================================================================
#  CONV W AVG POOL +1 FC, seq compressed to 63, channels to 100, FC 6300->500
# =============================================================================
class ConvNet(nn.Module):
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
