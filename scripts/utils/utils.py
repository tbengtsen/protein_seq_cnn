import os
import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from contextlib import redirect_stdout
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as cm
import datetime
        
####################################################################
##  Prepare data for Dataloader ##
####################################################################

class Prepare_Data(Dataset):

    def __init__(self, path, name, debug = False):
        '''Initializes the preprocessed Uniref dataset.
        Preprocessing prior to initialization cleans data. 
        See script preprocess_data.py 
        '''
        if debug:
            print('\n\nOBS! DEBUG MODE: only using 100 sequences per sub dataset')
            self.data = h5py.File(path, 'r').get(name)[:100]
        else:
            self.data = h5py.File(path, 'r').get(name)[:]

        self.X = torch.tensor(self.data, dtype=torch.long)   # must be long for onehot 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    
    
    
####################################################################
##  LOG all training parameters and model architecture
####################################################################

class LogFile():
    ''' creates log file
        - saves alle training and model params
        - logs when trained model are saved'''

    def __init__(self, options, model, working_dir):
        
        self.options = options # training script arguments
        self.model = model
        self.working_dir = working_dir
        
        self.log_file = self.create_log_file() 
        self.print_to_slurm(self.options)

    def create_log_file(self):
        '''creates log file ''' 
        logfile = self.working_dir + "logfile_params.log"
    
        if not os.path.isfile(logfile):
            with open(logfile, 'w') as f:
                self.write_log(f, self.options, self.model)

        # if file already exists, create with numbered name
        else:
            counter = 1
            logfile = self.working_dir + "logfile_params_{}.log"
            while os.path.isfile(logfile.format(counter)):
                    counter += 1
                    logfile = logfile.format(counter)
    
            with open(logfile, 'w') as f:
                self.write_log(f, self.options, self.model)
        
        return logfile

    def write_log(self, f, options, model):
        f.write('# Parameters for autoencoder \n' )
        f.write('# Training started on: {}\n'.format(datetime.date.today()) )
        f.write('\nTraining data file: {}\n'.format(options.data))
        f.write('\n\nTRAINING AND MODEL PARAMS:\n======================\n')

        for arg in vars(options):
            f.write('{} :\t\t {}\n'.format(arg, getattr(options, arg)))
        
        f.write('\n\nModel Summary:\n')
        with redirect_stdout(f):
            summary(model, input_size=(25, 500), batch_size=options.batch_size)

    def print_to_slurm(self, options):
        '''outputs params to slurm file. Not pretty, but efficient'''
        print('# Parameters for autoencoder compressing amino acids channels\n', flush=True )
        print('# Training started on: {}\n'.format(datetime.date.today()),flush=True )
        print('# All results are saved in : {}\n'.format(self.working_dir),flush=True)
        print('\nTRAINING AND MODEL PARAMS:\n')

        for arg in vars(options):
            print('{} :\t\t {}\n'.format(arg, getattr(options, arg)),flush=True)
         
        print('\nModel Summary:\n',flush=True)
        summary(self.model, input_size=(25, 500), batch_size=options.batch_size)

    def log_saved_model(self, steps):
        
        with open(self.log_file, 'a') as f:
            f.write('\nTraining model saved after training on {} sequences.'\
                    .format(steps*100) )
            
    def log_performance(self, acc, loss, ds_type='Train'):
        with open(self.log_file, 'a') as f:
            f.write('\n\n{} set loss: {}, accuracy: {}'.format(ds_type, loss, acc) )
            
            

        
        
####################################################################
##  Save model + optimizer states
####################################################################        
        
def save_checkpoint(model, optimizer, epoch, train_ds,loss_list, acc_list, working_dir):
    '''checkpoint file for restarting training so save everything 
    including optimizer state for adam optimizer'''
    
    state = {'working_dir' : working_dir,
             'epoch': epoch + 1, 
             'train_ds': train_ds+1, 
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), # important when using Adam opt
             'loss_list': loss_list,
             'acc_list': acc_list}
    
    torch.save(state, working_dir+'/checkpoint_model_epoch{}_train_ds{}.pt'.format(epoch,train_ds))

def load_checkpoint(model, optimizer, filename='XXX') :  
    '''Load cp model for restarting training. 
    Note: Input model & optimizer should be pre-defined. This routine only 
    updates their states. Due to saved as model.state_dict(). (Considered more safe.)
    See: 
    https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3 
    '''
    assert os.path.isfile(filename), f"=> no checkpoint found at '{filename}'"

    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']
    train_ds = checkpoint['train_ds']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_list = checkpoint['loss_list']
    acc_list = checkpoint['acc_list']
    #working_dir = checkpoint['working_dir']


#     print(f"=> loaded checkpoint '{filename}' - epoch {checkpoint['epoch']},\
#             train_ds{checkpoint['train_ds']}")

    # to cuda
    if torch.cuda.is_available():
        model.cuda()
        # all optim params to cuda
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        
        
    return model, optimizer, epoch, train_ds, loss_list, acc_list
                              
                              
def save_final_model(model, working_dir):
    '''Save final trained nn model and architechture'''
    torch.save(model, working_dir+'/model_final.pt')
    
    
def load_final_model(filename):
    '''load final trained nn model and architechture'''
    model = torch.load(filename)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    return model




####################################################################
## ONE HOT ENCODE 
####################################################################

def to_one_hot(X):
    """Embedding seq integers to one-hot form. """
    num_classes=25
    if torch.cuda.is_available():
        to_one_hot = torch.eye(num_classes,dtype=torch.float32,device='cuda')
    else:
        to_one_hot = torch.eye(num_classes,dtype=torch.float32)    
    
    return to_one_hot[X]


def from_one_hot(one_hot):
    """decode seq from one-hot form."""
    X = torch.argmax(one_hot,dim=1) 
    if  torch.cuda.is_available():
        X.cuda()  
    return X

# AA conversion to integers
aa_to_int = { 
            '0':0,# padding
            'M':1,
            'R':2,
            'H':3,
            'K':4,
            'D':5,
            'E':6,
            'S':7,
            'T':8,
            'N':9,
            'Q':10,
            'C':11,
            'U':12, # Selenocystein.
            'G':13,
            'P':14,
            'A':15,
            'V':16,
            'I':17,
            'F':18,
            'Y':19,
            'W':20,
            'L':21,
            'O':22, # Pyrrolysine
            'start':23,
            'stop':24 }

int_to_aa = {value:key for key, value in aa_to_int.items()}

def get_aa_to_int(aa):
    """
    Get the lookup table (for easy import)
    """

    return aa_to_int[aa]

def get_int_to_aa(i):
    """
    Get the lookup table (for easy import)
    """
    return int_to_aa[i]


triAA_to_int = {
            '0':0,# padding
            'MET':1,
            'ARG':2,
            'HIS':3,
            'LYS':4,
            'ASP':5,
            'GLU':6,
            'SER':7,
            'THR':8,
            'ASN':9,
            'GLN':10,
            'CYS':11,
            'SEC':12, # Selenocystein.
            'GLY':13,
            'PRO':14,
            'ALA':15,
            'VAL':16,
            'ILE':17,
            'PHE':18,
            'TYR':19,
            'TRP':20,
            'LEU':21,
            'PYL':22, # Pyrrolysine
            'start':23,
            'stop':24 }

int_to_triAA = {value:key for key, value in aa_to_int.items()}

def get_triAA_to_int(aa):
    """
    Get the lookup table (for easy import)
    """

    return triAA_to_int[aa]

def get_int_to_triAA(i):
    """
    Get the lookup table (for easy import)
    """
    return int_to_triAA[i]
