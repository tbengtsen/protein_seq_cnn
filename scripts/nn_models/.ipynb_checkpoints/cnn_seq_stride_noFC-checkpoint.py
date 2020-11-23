#!/usr/bin/python

#===============================================================
# Tone Bengtsen
# Creation Date: 08-10-2019
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


####################################################################
##  Input arguments   
####################################################################
# input arguments 
parser = argparse.ArgumentParser(description = 
        '\n \n \
         CNN AUTOENCODER TO COMPRESS SEQUENCE\n \
         ==================================\n \
         Takes preproccessed uniref file and\n \
         a set of model and training params \n \n', 
         formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument('--f', dest = "data", required=True, 
            help='Uniref preprocessed hd5f file. See preprocess_data.py')

parser.add_argument('--kernel_size', dest = 'kernel_size', 
        default=None, required = False, type=int,
	    help='Window size')

parser.add_argument('--stride', dest='stride',
        default=None, required=False, type=int,
        help ='Stride size for moving kernels/windows')

parser.add_argument ('--padding',dest='padding',
        default=None,required=False,  type=int,
	    help='padding on each sequence after each convolution.\n \
	    rule: -kernel+2P+1) = 0. E.g -5+2*2+1 =1 for k_size=5')

parser.add_argument ('--batch_size', dest='batch_size',
        default=100, required=False, type=int,
        help='Batch size for training' )

parser.add_argument ('--valid_size', dest='valid_size', 
        default=100, required=False, type=int,
	    help='Batch size for training' )

parser.add_argument('--learning_rate',dest='learning_rate',
        default = 0.0001, required=False,  type=float,
	    help='Learning rate for gradient')

parser.add_argument('--num_epochs',dest='num_epochs', 
        default=1,  type=int, required=False,
	    help='Number of epochs, i.e. iterations though training set')

parser.add_argument('--run_on_test',dest='testing', action="store_true",
	    help='Run trained model on test set? obs large!')

parser.add_argument('--debug',dest='debug', action="store_true", 
	    help='whether to only debug scripts - Loads only 1000 proteins for each data set')

####################################################################
##  Prepare data for Dataloader ##
####################################################################

class Prepare_Data(Dataset):

    def __init__(self, path, name):
        '''Initializes the preprocessed Uniref dataset.

        Preprocessing prior to initialization cleans data. 
        See script preprocess_data.py.

        '''
        self.X = h5py.File(path, 'r').get(name)[:]

        #self.X = np.load(path,allow_pickle=True )[name]

        self.X = torch.tensor(self.X,dtype=torch.long) # must be long for onehot    
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

####################################################################
##  Convolutional neural network (1 convolution layer) no pooling ##
####################################################################
class ConvNet(nn.Module):
    def __init__(self, k_size,stride,pad):
        '''test'''
        super().__init__()
        
        ## compress ##       
        self.conv1 = nn.Sequential(
            nn.Conv1d(25, 50,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv1a = nn.Sequential(
            nn.Conv1d(50, 50,  k_size, stride, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 125,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv3 = nn.Sequential(
            nn.Conv1d(125,125,  k_size, stride, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv4 = nn.Sequential(
            nn.Conv1d(125,125,  k_size, stride, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv5 = nn.Sequential(
            nn.Conv1d(125,125,  k_size, stride, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv6 = nn.Sequential(
            nn.Conv1d(125,125,  k_size, stride, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv7 = nn.Sequential(
            nn.Conv1d(125,125,  k_size, stride, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.conv8 = nn.Sequential(
            nn.Conv1d(125,125,  k_size, stride, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        
#         self.fc1_encode = nn.Sequential(
#             nn.Linear(100*63, 500),
#             nn.ReLU(),
#             nn.BatchNorm1d(500)) 
#         self.fc2_encode = nn.Sequential(
#             nn.Linear(1000, 500),
#             nn.ReLU(),
#             nn.BatchNorm1d(500)) 
    
        
        ## decompress ## 
#         self.fc2_decode = nn.Sequential(
#             nn.Linear(500, 1000),
#             nn.ReLU(),
#             nn.BatchNorm1d(1000)) 
#         self.fc1_decode = nn.Sequential(
#             nn.Linear(500, 100*63),
#             nn.ReLU(),
#             nn.BatchNorm1d(100*63))
        self.deconv8 = nn.Sequential(
            nn.Upsample(8, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv7 = nn.Sequential(
            nn.Upsample(16, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv6 = nn.Sequential(
            nn.Upsample(32, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv5 = nn.Sequential(
            nn.Upsample(63, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))

        self.deconv4 = nn.Sequential(
            nn.Upsample(125, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv3 = nn.Sequential(
            nn.Upsample(250, mode='linear', align_corners=True),
            nn.Conv1d(125,125,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(125))
        self.deconv2 = nn.Sequential(
            nn.Conv1d(125, 50,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1a = nn.Sequential(
            nn.Upsample(500, mode='linear', align_corners=True),
            nn.Conv1d(50, 50,  k_size, 1, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(50))
        self.deconv1 = nn.Sequential(
            nn.Conv1d(50, 25,  k_size, 1, padding=pad))
    

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
#         out = out.view(-1, out.shape[1]*out.shape[2]) 
#         out = self.fc1_encode(out)
        #out = self.fc2_encode(out)

        return out

    def decoder(self, out):
        #out = self.fc2_decode(out)
#         out = self.fc1_decode(out)
#         out = out.view(-1, 100,63) # re-shape array but not batches
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
 

####################################################################
##  TRAINING
####################################################################

def main():

    ## get model/training params 
    args = parser.parse_args()
    if args.debug:
        print ('==== DEBUGGING MODE ====')
    
    # get name of script for saving models
    script_name = os.path.basename(__file__)

    ## Initialize metrics  ###
    TrainingEval = utils.TrainingMetrics(script_name)
    working_dir  = TrainingEval.working_dir
    valid = Prepare_Data(args.data,'valid/valid')
    valid_batches = DataLoader(valid, args.batch_size, 
                               drop_last=True, shuffle=True)
    Validation = utils.Metrics(valid_batches, working_dir ,'validation')
    
    # cp running script to working dir. 
    os.system('cp {} {}'.format(script_name, working_dir))  

    
    ## Initialize model
    if torch.cuda.is_available(): 
        model = ConvNet(args.kernel_size, args.stride, args.padding).cuda()
    else:
        model = ConvNet(args.kernel_size, args.stride, args.padding) 
    
    ## log model/training params to file 
    LogFile = utils.LogFile(args, model, working_dir)
    
    ## Loss and optimizer 
    criterion = nn.CrossEntropyLoss() # doees not ignore padding (0) ignore_index=0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    
    # Train the model
    step = -1 # nr of batches 
    loss_list = []
    acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    
    #initiate random shuffle between  training dataset 
    random_ds = np.arange(0,20)
    np.random.shuffle(random_ds) # shuffle 
    
    for epoch in range(args.num_epochs):
        
        for train_ds in random_ds:
            f = args.data
            name = 'train/train_{}'.format(train_ds)
            train = Prepare_Data(f,name)
            train_batches = DataLoader(train, batch_size=args.batch_size, 
                               drop_last=True, shuffle=True)
           
            for i, batch in enumerate(train_batches):
                step += 1
    
                # one hot encode
                batch = utils.to_one_hot(batch)
    
                # transpose to input seq as vector
                batch = torch.transpose(batch,1,2) #transpose dim 1,2 => channels=aa
    
                ## Run the forward pass ##
                out = model(batch) # sandsynligheder=> skal være [10,25,502] hvor de 25 er sandsynligheder
                
                # convert back to aa labels from one hot for loss 
                batch_labels = utils.from_one_hot(batch) # integers for labels med 100% sikkerhed
    
    
                ## loss ##
                loss = criterion(out, batch_labels)
                loss_list.append(loss.item())
    
                ## switch model to training mode, clear gradient accumulators ##
                model.train()
                optimizer.zero_grad()
    
                ##  Backprop and perform Adam optimisation  ##
                loss.backward()
                optimizer.step()
    
                ##  Track the accuracy  ##
                if i  % 50 == 0:   
    #               ##########
                    acc = TrainingEval.get_acc(out,batch_labels)
                    acc_list.append(acc)
                    TrainingEval.save_metrics(acc, loss.item(), step, epoch)
                    print('Epoch [{}/{}], Step: {}, Loss: {:.4f}, Accuracy: {:.4f}%'
                            .format(epoch + 1, args.num_epochs, step, 
                                    loss.item(), acc*100))
    
                # Validation ##
                if i % 1000 == 0:
                    val_loss, val_acc, conf_matrix = \
                    Validation.get_performance(model,criterion,
                            confusion_matrix = True)
                    Validation.save(val_acc, val_loss, epoch, step)

                    # add to list for fast plotting
                    valid_loss_list.append(val_loss)
                    valid_acc_list.append(val_acc)
                    print('Validation:  Loss: {:.4f}, Accuracy: {:.4f}%\n'
                            .format(val_loss, val_acc*100))  
                    # plot 
                    TrainingEval.plot_metrics(acc_list, loss_list,
                                valid_acc_list, valid_loss_list, epoch)

                    Validation.plot_confusion_matrix(conf_matrix)
                    Validation.plot_per_class(conf_matrix)


            # Save the model every 5 train_-ds
            if train_ds % 5 ==0:
                utils.save_checkpoint(model, optimizer, epoch, train_ds,loss_list, acc_list, working_dir)
                utils.save_final_model(model, working_dir)
                LogFile.log_saved_model(step)
                LogFile.log_performance(acc, loss.item(), ds_type='Training')
    
                if args.testing: 

                    f = args.data
                    name = 'test/test_1'
                    test = Prepare_Data(f,name)
                    test_batches = DataLoader(test, batch_size=args.batch_size, 
                                       drop_last=True, shuffle=True)

                    Test = utils.Metrics(test_batches, working_dir ,'test')
                    test_loss, test_acc, conf_matrix = Test.get_performance(
                                        model, criterion, confusion_matrix = True)
                    Test.save(test_acc, test_loss, epoch=-1, step=-1)
                    Test.plot_confusion_matrix(conf_matrix)
                    Test.save_conf_matrix(conf_matrix)
                    Test.plot_per_class(conf_matrix)
                    LogFile.log_performance(test_acc, test_loss, ds_type='Test')
        
        
                
if __name__=="__main__":
    main()
