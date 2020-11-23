import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix as cm
from utils import Prepare_Data
from utils import to_one_hot
from utils import from_one_hot


##############################################################################
## Validation/test metrics class 
##############################################################################

class PerformMetrics():
    '''Calculates performance metrics on test or validation data dependent
    on object initialisation (ds_type='validation' or ='test')
    Arg:
        - data:  path to data
        - working dir: path to working dir for outputting
        - batch_size: for calculating performance
        - ds_type: dataset_type, either validation or test, only important 
        for naming
        
    returns: 
         - mean accuracy and/or loss given training model (+saves to file)
         - opt: calculate/plots confusion matrix
         - opt: calculate/plots position specific accuracy
    
    '''
    def __init__(self, data, working_dir, batch_size, ds_type='validation'):
        """ - Initializes object with either validation or test data batches 
            (from Dataloader).
            -  Calculates accuracy and/or loss given training
             model. 
             - Saves acc/loss to file
        """
        self.data = data 
        self.working_dir = working_dir
        self.batch_size = batch_size
        self.ds_type = self.assert_type(ds_type)
        self.fname = self.working_dir + '/{}_metrics.dat'.format(self.ds_type)
        self.create_f()
        self.aa_to_int = { 
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
        self.int_to_aa = {value:key for key, value in self.aa_to_int.items()}
    


    def assert_type(self, ds_type):
        '''check if valid data type string 
        '''
        
        assert ds_type in ['validation','test','valid'], 'Class Metrics \
            attribute: ds_type must be either "validation" or "test"'

        if ds_type is 'validation': ds_type = 'valid'
        
        return ds_type
        
        
    def create_f(self,):
        '''Creates file for saving acc/loss in 
        '''

        with open(self.fname, 'w') as f:
            f.write('# CNN to encode/compress amino acids. \n')
            f.write('# Created on: {}\n'.format(datetime.date.today().ctime() ))
            f.write('# epoch  nr_used_batches  loss  accuracy padding_accuracy \n')
        
        
    def save(self, acc, loss, pad_acc, epoch, step):
        '''Saves acc/loss to file 
        '''
        with open(self.fname, 'a') as f:
            f.write('{}\t {}\t {:.4f}\t {:.4f} \t {:.4f} \n'.format(
                            epoch + 1, step, loss, acc*100, pad_acc*100))
        
        
    def get_performance(self,
                        model,
                        criterion, 
                        confusion_matrix = False, 
                        pos_acc = False, 
                        debug = False): 
        ''' Get mean accuracy of model from init data/batches
        '''
        
        
        model.eval() # with eval mode, there is large udsving i validation, due to params updated faster than in train mode

        ## init performance stuff ##
        acc_list = [] # list for correct predictions
        loss_list = []
        acc_list_pad =[] # list for predicting paddings
        conf_matrix = np.zeros((25,25))
        
        ## position specific accuracy 
        N_term_pos = torch.from_numpy(np.zeros((500), dtype=np.int))
        N_term_pad = torch.from_numpy(np.zeros((500), dtype=np.int)) # correctness of padding
        C_term_pos= torch.from_numpy(np.zeros((20), dtype=np.int))
        # get len of proteins for normalisation of posiiton accuraty
        len_seq = torch.from_numpy(np.zeros((500), dtype=np.int)) # for normalisation
        len_pad = torch.from_numpy(np.zeros((500), dtype=np.int))
        len_seq_C_term = 0.0 # for normalisation - does not need to seee how often, as always same
        

        # initiate random shuffle between sub dataset (ds)
        h5_file = h5py.File(self.data , 'r')
        random_ds = list(h5_file[self.ds_type].keys())
        random_ds = np.array(random_ds)
        np.random.shuffle(random_ds) # shuffle

        # loop over all datasets of that datatype (test/valid)
        for idx, ds in enumerate(random_ds):

            ds = str(self.ds_type) + '/' + ds
            prep_data = Prepare_Data(path=self.data, name=ds, debug=debug)
            # inialise from dataloader
            batches = DataLoader(prep_data, self.batch_size, 
                                   drop_last=True, shuffle=True)

            with torch.no_grad():
                for i, batch in enumerate(batches):
                    batch = to_one_hot(batch)

                    # transpose to input seq as vector
                    batch = torch.transpose(batch,1,2) #transpose dim 1,2 => channels=aa

                    ## Run the forward pass ##
                    out = model(batch) # sandsynligheder=> skal vaere [10,25,502] hvor de 25 er sandsynligheder

                    # convert back to aa labels from one hot for loss 
                    batch_labels = from_one_hot(batch) # integers for labels med i.e. 100% sikkerhed

                    # loss 
                    loss = criterion(out, batch_labels)
                    loss_list.append(loss.item())

                    # get accuracy 
                    _, predicted = torch.max(out.data, dim=1) # convert back from one hot
                    
                    ## PREDICTIONS/ACCURACY ##
                    # filter out padding (0)
                    msk = batch_labels != 0
                    target_msk = batch_labels[msk]
                    pred_msk = predicted[msk]
                     
                    # count correct predictions
                    total = target_msk.shape[0]
                    correct = (pred_msk == target_msk).sum().item()

                    # save to list 
                    acc_list.append(correct / total)
                    
                    ## PADDINGS PREDICTIONS ##
                    # filter out all but padding (=0)
                    msk_pad = batch_labels == 0
                    target_msk_pad = batch_labels[msk_pad]
                    pred_msk_pad = predicted[msk_pad]
                    # count correct predictions
                    total_pad = target_msk_pad.shape[0]
                    correct_pad = (pred_msk_pad == target_msk_pad).sum().item()

                    # save to list 
                    acc_list_pad.append(correct_pad / total_pad)
                    

                    # confusion matrix
                    if confusion_matrix :
                        conf_matrix += cm(
                        target_msk.view(-1).cpu(), 
                        pred_msk.view(-1).cpu(),
                        labels = np.arange(25) )


                    # get position specific accuracy
                    if pos_acc:
                        # N-Term
                        correct = (batch_labels == predicted)
                        correct_msk = np.logical_and(correct.cpu(), msk.cpu()) #msk out padding
                        N_term_pos += torch.sum(correct_msk, dim=0) # sum ovre columns 
                        len_seq  +=  torch.sum(msk.cpu(), dim=0)
                        # Padding N-Term 
                        correct_msk_pad = np.logical_and(correct.cpu(), msk_pad.cpu()) #msk out all but padding
                        N_term_pad += torch.sum(correct_msk_pad, dim=0) # sum ovre columns 
                        len_pad  +=  torch.sum(msk_pad.cpu(), dim=0) # sum how often pad correct at given pos
                        
                        # C-term
                        #find end index of last aa for each protein, for predictions in C-term
                        bckwrds_indx = torch.sum(msk, dim=1) 
                        bckwrds_indx += -21 

                        try: # in case protein smaller than 21
                            bckwrds = self.bck_seq(correct_msk, bckwrds_indx.cpu(), num_elem=20)
                            C_term_pos += torch.sum(bckwrds , dim=0)
                            len_seq_C_term += self.batch_size

                        except: 
                            pass



        # average accuracy on test set        
        mean_acc = sum(acc_list)/len(acc_list)
        mean_acc_pad = sum(acc_list_pad)/len(acc_list_pad)
        mean_loss = sum(loss_list)/len(loss_list)
        model.train()
        
        if pos_acc:        
            # position specific accuracy        
            N_term = np.divide(N_term_pos[:],len_seq[:])
            C_term = C_term_pos/ float(len_seq_C_term)
            N_pad = np.divide(N_term_pad[:],len_pad[:])  

        # return things
        if confusion_matrix and pos_acc: 
            return mean_loss, mean_acc, mean_acc_pad, conf_matrix, N_term, C_term, N_pad
        
        elif confusion_matrix and not pos_acc: 
            return mean_loss, mean_acc, mean_acc_pad, conf_matrix
        
        elif not confusion_matrix and pos_acc: 
            return mean_loss, mean_acc, mean_acc_pad, N_term, C_term, N_pad
        
        else: 
            return mean_loss, mean_acc, mean_acc_pad
    
    
    def bck_seq(self, matrix,  indx, num_elem=20):
        '''takes a matrix and return new matrix of shape: m,n = A.shape[0], num_elem
        Where each row corresponds to  A but with len=num_elem and starting from the index defined in indx '''
        all_indx = indx[:,None] + torch.arange(num_elem)
        return matrix[np.arange(all_indx.shape[0])[:,None], all_indx]

    def plot_pos_acc(self, N_term, C_term, N_pad):
        # plot N-term accuracy 
        fig = plt.figure(figsize=[12,2])
        plt.bar(np.arange(0,500) ,N_term) #plt.cm.RdYlGn
        plt.grid()
        plt.title('N-term positional accuracy')
        plt.ylim(0.0,1.0)
        plt.xlabel('position in sequence')
        plt.savefig(self.working_dir+'/plot_pos_acc_N-term.png')
        plt.close('all')

        # plt.tight_layout()
        # plot C-term accuracy
        tick_marks = np.arange(1,21)
        pos_c_term = [i-21 for i in tick_marks ] # count bckwards
        fig = plt.figure(figsize=[12,2])
        plt.bar(range(1,21),C_term) #plt.cm.RdYlGn
        plt.ylim(0.0,1.0)
        plt.grid()
        plt.title('C-term positional accuracy')
        plt.xlabel('position in sequence from C-terminal ')
        plt.xticks(tick_marks, pos_c_term)
        plt.savefig(self.working_dir+'/plot_pos_acc_C-term.png')
        plt.close('all')
        
        # plot padding accuracy 
        fig = plt.figure(figsize=[12,2])
        plt.bar(np.arange(0,500), N_pad) #plt.cm.RdYlGn
        plt.grid()
        plt.title('Padding positional accuracy')
        plt.ylim(0.0,1.0)
        plt.xlabel('position in sequence')
        plt.savefig(self.working_dir+'/plot_pos_acc_padding.png')
        plt.close('all')
        
    
    def plot_confusion_matrix(self, conf_matrix ):
        ''' plot and save confusion matrix'''
        cm = conf_matrix # easier ... 
        
        # delete start/stop/O,U in both row and columns
        cm = np.delete(cm,[0,12,22,23,24],0) 
        cm = np.delete(cm,[0,12,22,23,24],1)
        #swap L and F so L close to V,I
        cm[:,[19,16]] = cm[:,[16,19]] 
        cm[[19,16],:] = cm[[16,19],:]
        #swap F and W
        cm[:,[19,18]] = cm[:,[18,19]]  
        cm[[19,18],:] = cm[[18,19],:]
        #swap G and P
        cm[:,[11,12]] = cm[:,[12,11]]  
        cm[[11,12],:] = cm[[12,11],:]
        
        # normalise taken from scikitlearn
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]        
        # plot 
        fig = plt.figure(figsize=[9,9])
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Spectral)
        plt.title('Confusion matrix - normalised', fontsize=20)
        plt.colorbar()
    
        # plot settings
        tick_marks = np.arange(20)
        classes = [self.int_to_aa[i] for i in np.arange(25)]
        classes = np.delete(classes,[0,12,22,23,24],0)
        classes[[19,16]] = classes[[16,19]] #swap L and F so L close to V,I
        classes[[19,18]] = classes[[18,19]] #swap F and W 
        classes[[11,12]] = classes[[12,11]] #swap G and P 

        
        plt.xticks(tick_marks, classes, fontsize=14, rotation=45)
        plt.yticks(tick_marks, classes, fontsize=14)
        
        plt.tight_layout()
        plt.ylabel('True label',fontsize=16)
        plt.xlabel('Predicted label',fontsize=16)
    
        plt.savefig(self.working_dir+'/plot_confusion_matrix_{}.png'.format(self.ds_type))
        plt.close('all')
        
        
    def save_conf_matrix(self, conf_matrix):
        ''' saves confusion matrix in working dir'''
        
        np.save(self.working_dir+'/confusion_matrix.npy', conf_matrix)
        
    def per_class_accuracy(self, conf_matrix):
        '''accuracy per class/aa '''    
        cm = conf_matrix
        per_class_acc = cm.diagonal()/cm.sum(axis=1)

        return per_class_acc
    
    
    def plot_per_class(self, conf_matrix):
        '''plot and save per class (per aa) accuracy'''
        cm = conf_matrix
        
        # delete start/stop/O,U in both row and columns
        cm = np.delete(cm,[0,12,22,23,24],0) 
        cm = np.delete(cm,[0,12,22,23,24],1)
        #swap L and F so L close to V,I
        cm[:,[19,16]] = cm[:,[16,19]] 
        cm[[19,16],:] = cm[[16,19],:]
        #swap F and W
        cm[:,[19,18]] = cm[:,[18,19]]  
        cm[[19,18],:] = cm[[18,19],:]
        #swap G and P
        cm[:,[11,12]] = cm[:,[12,11]]  
        cm[[11,12],:] = cm[[12,11],:]
        
        # normalise per column - taken from scikitlearn        
        per_class_acc = cm.diagonal()/cm.sum(axis=1)
        
        # define labels 
        classes = [self.int_to_aa[i] for i in np.arange(25)]
        classes = np.delete(classes,[0,12,22,23,24],0)
        classes[[19,16]] = classes[[16,19]] #swap L and F so L close to V,I
        classes[[19,18]] = classes[[18,19]] #swap F and W 
        classes[[11,12]] = classes[[12,11]] #swap G and P 

        # plot
        fig = plt.figure(figsize=[9,9])
        plt.bar(classes, height=per_class_acc)
        
        # plot settings
        tick_marks = np.arange(20)
        plt.xticks(tick_marks, classes, fontsize=14,rotation=45)
        plt.yticks(fontsize=14)
        plt.title('Per amino acid accuracy',fontsize=20)

        # total of each aa written on the top of each bar
        for i in np.arange(20):
            plt.text(i-0.8, y = per_class_acc[i]+0.01,
                     s = '{0:.1}'.format(cm.sum(axis=1)[:, np.newaxis][i][0]),
                     size = 12, rotation=45)
        
        plt.savefig(self.working_dir+'/plot_acc_per_aa_{}.png'.format(self.ds_type))
        
        plt.close('all') 



### ++++++ OLD +++++ ###
class Metrics_OLD():
        
    def __init__(self, batches, working_dir , ds_type='validation'):
        """ - Initializes object with either validation or test data batches 
            (from Dataloader).
            -  Calculates accuracy and/or loss given training
             model. 
             - Saves acc/loss to file
        """
        self.batches = batches 
        self.working_dir = working_dir
        self.ds_type = ds_type
        assert self.ds_type is 'validation' or  self.ds_type is 'test',\
            'Class Metrics attribute: ds_type must be either "validation" or "test"'

        self.fname = self.working_dir + '/{}_metrics.dat'.format(self.ds_type)
        self.create_f()
        self.aa_to_int = { 
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
        self.int_to_aa = {value:key for key, value in self.aa_to_int.items()}

    
    def create_f(self,):
        '''creates file for saving acc/loss in '''
        with open(self.fname, 'w') as f:
            f.write('# CNN to encode/compress amino acids. \n')
            f.write('# Created on: {}\n'.format(datetime.date.today().ctime() ))
            f.write('# epoch  nr_used_batches  loss  accuracy \n')
        
        
    def save(self, acc, loss, epoch, step):
        '''Saves acc/loss to file '''
        with open(self.fname, 'a') as f:
            f.write('{}\t {}\t {:.4f}\t {:.4f} \n'.format(
                     epoch + 1, step, loss, acc*100))
        
        
    def get_performance(self, model, criterion, confusion_matrix = False, pos_acc = False ): 
        '''get mean accuracy of model from init data/batches'''
        
        # performance stuff
        model.eval() # with eval mode, there is large udsving i validation, due to params updated faster than in train mode
        acc_list = []
        loss_list = []
        conf_matrix = np.zeros((25,25))
        
        ## pos specific accuracy 
        N_term_pos = torch.from_numpy(np.zeros((500), dtype=np.int))
        C_term_pos= torch.from_numpy(np.zeros((20), dtype=np.int))
        len_seq = torch.from_numpy(np.zeros((500), dtype=np.int)) # for normalisation
        len_seq_C_term = 0.0 # for normalisation - does not need to seee how often, as always same
        

        with torch.no_grad():
            for i, batch in enumerate(self.batches):
                batch = to_one_hot(batch)

                # transpose to input seq as vector
                batch = torch.transpose(batch,1,2) #transpose dim 1,2 => channels=aa

                ## Run the forward pass ##
                out = model(batch) # sandsynligheder=> skal vaere [10,25,502] hvor de 25 er sandsynligheder
                
                # convert back to aa labels from one hot for loss 
                batch_labels = from_one_hot(batch) # integers for labels med i.e. 100% sikkerhed
                
                # loss 
                loss = criterion(out, batch_labels)
                loss_list.append(loss.item())
        
                # get accuracy 
                _, predicted = torch.max(out.data, dim=1) # convert back from one hot

                # filter out padding (0)
                msk = batch_labels != 0
                target_msk = batch_labels[msk]
                pred_msk = predicted[msk]

                # count correct predictions
                total = target_msk.shape[0]
                correct = (pred_msk == target_msk).sum().item()
                incorrect = (pred_msk != target_msk).sum().item()
               
                # save to list 
                acc_list.append(correct / total)
                
                # confusion matrix
                if confusion_matrix :
                    conf_matrix += cm(
                    target_msk.view(-1).cpu(), 
                    pred_msk.view(-1).cpu(),
                    labels = np.arange(25) )
                
                
                # get position specific accuracy
                if pos_acc:
                    correct = (batch_labels == predicted)
                    correct_msk = np.logical_and(correct.cpu(), msk.cpu()) #msk out padding
                    N_term_pos += torch.sum(correct_msk, dim=0) # sum ovre columns 
                    len_seq  +=  torch.sum(msk.cpu(), dim=0)

                    bckwrds_indx = torch.sum(msk, dim=1) #find end indeex of each protein
                    bckwrds_indx += -21 

                    try: # in case protein smaller than 21
                        bckwrds = self.bck_seq(correct_msk, bckwrds_indx.cpu(), num_elem=20)
                        C_term_pos += torch.sum(bckwrds , dim=0)
                        len_seq_C_term += batch_size

                    except: 
                        pass


                
        # average accuracy on test set        
        mean_acc = sum(acc_list)/len(acc_list)
        mean_loss = sum(loss_list)/len(loss_list)
        model.train()
        
        if pos_acc:        
            # position specific accuracy        
            N_term = np.divide(N_term_pos[:],len_seq[:])
            C_term = C_term_pos/ float(len_seq_C_term)

        # return things
        if confusion_matrix and pos_acc: return mean_loss, mean_acc, conf_matrix, N_term, C_term
        elif confusion_matrix and not pos_acc: return mean_loss, mean_acc, conf_matrix
        elif not confusion_matrix and pos_acc: return mean_loss, mean_acc, N_term, C_term
        else: return mean_loss, mean_acc
    
    
    def bck_seq(self, matrix,  indx, num_elem=20):
        '''takes a matrix and return new matrix of shape: m,n = A.shape[0], num_elem
        Where each row corresponds to  A but with len=num_elem and starting from the index defined in indx '''
        all_indx = indx[:,None] + torch.arange(num_elem)
        return matrix[np.arange(all_indx.shape[0])[:,None], all_indx]

    def plot_pos_acc(self, N_term, C_term):
        fig = plt.figure(figsize=[12,2])
        plt.bar(np.arange(0,500) ,N_term) #plt.cm.RdYlGn
        plt.grid()
        plt.title('N-term positional accuracy')
        plt.ylim(0.0,1.0)
        plt.xlabel('position in sequence')
        plt.savefig(self.working_dir+'/plot_pos_acc_N-term.png')
        plt.close('all')

        # plt.tight_layout()
        # C-trm
        tick_marks = np.arange(1,21)
        pos_c_term = [i-21 for i in tick_marks ] # count bckwards
        fig = plt.figure(figsize=[12,2])
        plt.bar(range(1,21),C_term) #plt.cm.RdYlGn
        plt.ylim(0.0,1.0)
        plt.grid()
        plt.title('C-term positional accuracy')
        plt.xlabel('position in sequence from C-terminal ')
        plt.xticks(tick_marks, pos_c_term)
        plt.savefig(self.working_dir+'/plot_pos_acc_C-term.png')
        plt.close('all')
    
    
    def plot_confusion_matrix(self, conf_matrix ):
        ''' plot and save confusion matrix'''
        cm = conf_matrix # easier ... 
        
        # delete start/stop/O,U in both row and columns
        cm = np.delete(cm,[0,12,22,23,24],0) 
        cm = np.delete(cm,[0,12,22,23,24],1)
        #swap L and F so L close to V,I
        cm[:,[19,16]] = cm[:,[16,19]] 
        cm[[19,16],:] = cm[[16,19],:]
        #swap F and W
        cm[:,[19,18]] = cm[:,[18,19]]  
        cm[[19,18],:] = cm[[18,19],:]
        #swap G and P
        cm[:,[11,12]] = cm[:,[12,11]]  
        cm[[11,12],:] = cm[[12,11],:]
        
        # normalise taken from scikitlearn
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]        
        # plot 
        fig = plt.figure(figsize=[9,9])
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Spectral)
        plt.title('Confusion matrix - normalised', fontsize=20)
        plt.colorbar()
    
        # plot settings
        tick_marks = np.arange(20)
        classes = [self.int_to_aa[i] for i in np.arange(25)]
        classes = np.delete(classes,[0,12,22,23,24],0)
        classes[[19,16]] = classes[[16,19]] #swap L and F so L close to V,I
        classes[[19,18]] = classes[[18,19]] #swap F and W 
        classes[[11,12]] = classes[[12,11]] #swap G and P 

        
        plt.xticks(tick_marks, classes, fontsize=14, rotation=45)
        plt.yticks(tick_marks, classes, fontsize=14)
        
        plt.tight_layout()
        plt.ylabel('True label',fontsize=16)
        plt.xlabel('Predicted label',fontsize=16)
    
        plt.savefig(self.working_dir+'/plot_confusion_matrix_{}.png'.format(self.ds_type))
        plt.close('all')
        
        
    def save_conf_matrix(self, conf_matrix):
        ''' saves confusion matrix in working dir'''
        
        np.save(self.working_dir+'/confusion_matrix.npy', conf_matrix)
        
    def per_class_accuracy(self, conf_matrix):
        '''accuracy per class/aa '''    
        cm = conf_matrix
        per_class_acc = cm.diagonal()/cm.sum(axis=1)

        return per_class_acc
    
    
    def plot_per_class(self, conf_matrix):
        '''plot and save per class (per aa) accuracy'''
        cm = conf_matrix
        
        # delete start/stop/O,U in both row and columns
        cm = np.delete(cm,[0,12,22,23,24],0) 
        cm = np.delete(cm,[0,12,22,23,24],1)
        #swap L and F so L close to V,I
        cm[:,[19,16]] = cm[:,[16,19]] 
        cm[[19,16],:] = cm[[16,19],:]
        #swap F and W
        cm[:,[19,18]] = cm[:,[18,19]]  
        cm[[19,18],:] = cm[[18,19],:]
        #swap G and P
        cm[:,[11,12]] = cm[:,[12,11]]  
        cm[[11,12],:] = cm[[12,11],:]
        
        # normalise per column - taken from scikitlearn        
        per_class_acc = cm.diagonal()/cm.sum(axis=1)
        
        # define labels 
        classes = [self.int_to_aa[i] for i in np.arange(25)]
        classes = np.delete(classes,[0,12,22,23,24],0)
        classes[[19,16]] = classes[[16,19]] #swap L and F so L close to V,I
        classes[[19,18]] = classes[[18,19]] #swap F and W 
        classes[[11,12]] = classes[[12,11]] #swap G and P 

        # plot
        fig = plt.figure(figsize=[9,9])
        plt.bar(classes, height=per_class_acc)
        
        # plot settings
        tick_marks = np.arange(20)
        plt.xticks(tick_marks, classes, fontsize=14,rotation=45)
        plt.yticks(fontsize=14)
        plt.title('Per amino acid accuracy',fontsize=20)

        # total of each aa written on the top of each bar
        for i in np.arange(20):
            plt.text(i-0.8, y = per_class_acc[i]+0.01,
                     s = '{0:.1}'.format(cm.sum(axis=1)[:, np.newaxis][i][0]),
                     size = 12, rotation=45)
        
        plt.savefig(self.working_dir+'/plot_acc_per_aa_{}.png'.format(self.ds_type))
        
        plt.close('all') 


##############################################################################
## Training metrics class 
##############################################################################

class TrainingMetrics():
    """ Calculates and saves training accuracy and loss as well as nn model"""
    #def __init__(self, script_name):
    def __init__(self, top_working_dir, restart_from=None):
#        self.top_working_dir = top_working_dir
#         self.script_name = script_name
        self.working_dir = self._make_working_dir(top_working_dir, restart_from)
        self.fname = self.working_dir+'training_metrics.dat' 
        self._create_metrics_file()


    def get_acc(self, out, batch_labels): 
        '''get accuracy of model during training '''

        # get accuracy 

        _, predicted = torch.max(out.data, dim=1) # convert back from one hot

        # filter out padding (0)
        msk = batch_labels != 0
        target_msk = batch_labels[msk]
        pred_msk = predicted[msk]

        # count correct predictions
        total = target_msk.shape[0]
        correct = (pred_msk == target_msk).sum().item()
        incorrect = (pred_msk != target_msk).sum().item()

        acc = correct / total

        return acc 

    
    def _make_working_dir(self, top_working_dir, restart_from):
        ''' creates directory for saving trained model and related data
        '''

        # create top dir named after nn model
        top_dir = '../models/'+ top_working_dir

        # Debugging mode, create dir in specific debugging dir
        if top_working_dir is 'debugging':
            # create subdir w name of date of submission     
            working_dir = self._create_date_subdir(top_dir)
        
        
        # if restart from model:
        elif restart_from is not None:
            # get directory name from path to checkpoint model 
            restart_dir = restart_from.split('/checkpoint_model')[0]
            working_dir = restart_dir + '/Restart'
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)

        # if not restart, and no o_dir specified, create new dir name of
        # nn_model and subdir with name of submission data
        else: 
            if not os.path.exists(top_dir):
                os.makedirs(top_dir)
            # name subdir after date of submission     
            working_dir = self._create_date_subdir(top_dir) 

        return working_dir + '/'


    def _create_date_subdir(self,top_dir):
        ''' name subdir after date of submission '''
        
        working_dir = top_dir+'/{}'.format(\
                            datetime.date.today()) 
        # check if sub directory exists or create new with number
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        else:
            counter = 1
            working_dir = working_dir + '_{}'
            while os.path.exists(working_dir.format(counter)):
                counter += 1
            working_dir = working_dir.format(counter)           
            os.makedirs(working_dir)

        return working_dir 


    def _create_metrics_file(self):
        '''initiates file for saving training metrics '''
        with open(self.fname, 'w') as f:
            f.write('# CNN to encode/compress amino acids.\n')
            f.write('# Created on: {} \n'.format(datetime.date.today().ctime() ))
            f.write('# epoch  nr_used_batches  loss  accuracy \n')


    def save_metrics(self, acc, loss, step, epoch):
        '''Saves training metrics'''
        with open(self.fname, 'a+') as f:
            f.write('{}\t {}\t {:.4f}\t {:.4f} \n'.format(
                epoch + 1, step, loss, acc*100))
    
    def save_model(self, model, batch_id):
        '''Save training nn model'''
        torch.save(model, self.working_dir+'/model_CNN_{}.pt'.format(batch_id))

        
    def plot_metrics(self, acc_list, loss_list, valid_acc_list,
                            valid_loss_list, epoch ):

        ''' plt batch vs acc and batch vs loss '''
        fig = plt.figure(figsize=[15,7])
        ax1 = fig.add_subplot(121)
        scale = int(len(acc_list)/len(valid_acc_list))
        ax1.plot(np.arange(len(acc_list))*50, acc_list, label='training accuracy')
        ax1.plot(np.arange(len(valid_acc_list))*scale*50, valid_acc_list, label='validation accuracy')
        ax1.legend(fontsize=16)
        ax1.set_xlabel('batches',fontsize=16)
        ax1.set_title('Training after {} epochs'.format(epoch),fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        
        ax2 = fig.add_subplot(122)
        ax2.plot(np.arange(len(loss_list))[::50], loss_list[::50],label='training loss')
#         ax2.plot(np.arange(len(valid_loss_list))*scale, valid_loss_list, label='validation loss')
        ax2.set_xlabel('batches',fontsize=16)
        ax2.legend(fontsize=16) 
        ax2.tick_params(axis='both', which='major', labelsize=14)
        fig.savefig(self.working_dir+'plot_training_performance.png', bbox_inches='tight')

