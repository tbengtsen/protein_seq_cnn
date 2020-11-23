#===============================================================
# Tone Bengtsen
# Creation Date: 27-11-2019
#===============================================================
import torch
import torch.nn as nn
    
# =============================================================================
#       CONV W AVG POOL no FC, seq compressed to 8, channels to 50
# =============================================================================


class ConvNet(nn.Module):
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
