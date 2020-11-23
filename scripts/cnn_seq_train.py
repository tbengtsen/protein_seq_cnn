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
from torch.utils.data import DataLoader #Dataset, DataLoader
import utils
from contextlib import redirect_stdout
import importlib

####################################################################
##  Input arguments   
####################################################################
# input arguments 
parser = argparse.ArgumentParser(description = 
        '\n \n \
         CNN AUTOENCODER TO COMPRESS SEQUENCE\n \
         ==================================\n \
         Takes preproccessed uniref file, \n \
         a model and training params \n \n', 
         formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument('--f', dest = "data", required=True, 
        help='Uniref preprocessed hd5f file. See preprocess_data.py')

parser.add_argument('--out_dir', dest = "out_dir",
        required=False, default=None,    
        help='Name of output directory for dumping all files of this program.\
        If not specified, the program will create a directory named after the\
        nn model and then a sub-directory with named as the submission date ')

parser.add_argument('--model', dest = "nn_model", required=True, 
        help='Name of nn model (class). Must be defined in scripts in \
        directory nn_models/ and imported in __init__.py\
        in that directory.')

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

parser.add_argument('--ks_pool', dest = 'ks_pool', 
        default=None, required = False, type=int,
	    help='Window size for avg pooling')

parser.add_argument('--str_pool', dest='str_pool',
        default=None, required=False, type=int,
        help ='Stride size for moving kernels/windows in pooling')

parser.add_argument ('--pad_pool',dest='pad_pool',
        default=None,required=False,  type=int,
	    help='padding on each sequence after each avg pool.\n \
	    rule: -kernel+2P+1) = 0. E.g -5+2*2+1 =1 for k_size=5')

parser.add_argument ('--batch_size', dest='batch_size',
        default=100, required=False, type=int,
        help='Batch size for training' )


parser.add_argument('--learning_rate',dest='learning_rate',
        default = 0.0001, required=False,  type=float,
	    help='Learning rate for gradient')

parser.add_argument('--num_epochs',dest='num_epochs', 
        default=2,  type=int, required=False,
	    help='Number of epochs, i.e. iterations though training set')

parser.add_argument('--run_on_test',dest='testing', action="store_true",
	    help='Run trained model on test set? obs large!')

parser.add_argument('--restart_from',dest='restart',
        required=False, default = None,         
	    help='path to pt model file to restart from. Must be saved as in utils')

parser.add_argument('--debug',dest='debug', action="store_true", 
	    help='whether to only debug scripts - Loads only 1000 proteins for each data set')

####################################################################
##  TRAINING
####################################################################
def main():

    ## get model/training params ##
    args = parser.parse_args()

    ## specify name of output dir ##
    # dir to be created once initializing TrainingMetrics
    if args.debug:
        top_working_dir = 'debugging'

    elif args.out_dir is not None:
        top_working_dir = args.out_dir
    
    else:
        top_working_dir = str(args.nn_model.split(".py")[0]) 
        
    
    ## Initialize training metrics  ###
    
    # simultanously creates working_dir 
    TrainingEval = utils.TrainingMetrics(top_working_dir, 
                                         args.restart) 

    # get name of output/working dir
    working_dir = TrainingEval.working_dir

    
    ## Initialize Validation metrics ##
    Validation = utils.PerformMetrics(args.data, 
                                        working_dir,
                                        args.batch_size, 
                                        'validation')
    
    
    ## Initialise Test metrics: ##
    if args.testing: 
        Test = utils.PerformMetrics(args.data, 
                                     working_dir, 
                                     args.batch_size, 
                                     'test')
    

    ## Logging of scripts, models and params ##
    # cp nn_model script to working dir. 
    os.system('cp nn_models/{} {}'.format(args.nn_model, working_dir))  

    ## Load nn model architecture ##
    path = './nn_models/' + args.nn_model 
    spec = importlib.util.spec_from_file_location('nn_module', path)
    nn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nn_module)
    model = nn_module.ConvNet(args.kernel_size, args.stride, args.padding,
                         args.ks_pool, args.str_pool, args.pad_pool)

#     nn_model = importlib.import_module('.{}'.format(args.nn_model), package='nn_models')
#     model = nn_model.ConvNet(args.kernel_size, args.stride, args.padding,
#                              args.ks_pool, args.str_pool, args.pad_pool)  
    # CUDA 
    if torch.cuda.is_available(): 
        model = model.cuda()

    # initalise optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # load from restart file, params are conv to cuda in loading
    if args.restart is not None:
        model, optimizer, epoch_start, train_idx, loss_list, acc_list = \
            utils.load_checkpoint(model, optimizer, filename=args.restart) 
        print('loaded checkpoint model', flush=True)
    else:
        loss_list = []
        acc_list = []
        epoch_start = 0 



    # log model/training params to file 
    LogFile = utils.LogFile(args, model, working_dir)

    ## Loss 
    criterion = nn.CrossEntropyLoss() # does not ignore padding (0) ignore_index=0


    # Train the model
    nr_of_batches = -1 # count batches for logging
    valid_loss_list = []
    valid_acc_list = []

    #initiate random shuffle between  training sub dataset
    random_ds = list(h5py.File(args.data,'r')['train'].keys()) # get sub-names 
    random_ds = np.array(random_ds)
    np.random.shuffle(random_ds) # shuffle 
    
    # loop over entire training set multiple times
    for epoch in range(epoch_start, args.num_epochs):

        # loop over sub training sets (for memory reasons)
        for train_idx, sub_name in enumerate(random_ds):
            # load data 
            f = args.data
            name = 'train/{}'.format((sub_name))
            train = utils.Prepare_Data(f, name, debug=args.debug)

            # make batches of the data
            train_batches = DataLoader(train, batch_size=args.batch_size, 
                               drop_last=True, shuffle=True)

            for i, batch in enumerate(train_batches):
                nr_of_batches+= 1

                # one hot encode
                batch = utils.to_one_hot(batch)

                # transpose to input seq as vector
                batch = torch.transpose(batch,1,2) #transpose dim 1,2 => channels=aa

                ## Run the forward pass ##
                out = model(batch) # sandsynligheder=> skal vaere [10,25,502] hvor de 25 er sandsynligheder

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

            ##  Track the training accuracy  ##
            if train_idx  % 1 == 0:   
                acc = TrainingEval.get_acc(out,batch_labels)
                acc_list.append(acc)
                TrainingEval.save_metrics(acc, loss.item(), nr_of_batches, epoch)
                print('Epoch [{}/{}], sub training set: {} , nr_batches: {}, Loss: {:.4f}, Accuracy: {:.4f}%'
                        .format(epoch, args.num_epochs, train_idx, nr_of_batches, 
                                loss.item(), acc*100), flush=True)

                # Validation ##
           #    # if i % 1000 == 0:
            if train_idx % 5 == 0:     
                # get nn model performance on valid set
                val_loss, val_acc, val_acc_pad,  N_term,C_term, N_pad = Validation.get_performance(
                        model,criterion,  pos_acc=True ,debug=args.debug)
                
                # save validation metrics to file
                Validation.save(val_acc, val_loss, val_acc_pad, epoch, nr_of_batches)

                # add to list for fast plotting
                valid_loss_list.append(val_loss)
                valid_acc_list.append(val_acc)
                print('Validation:  Loss: {:.4f}, Accuracy: {:.4f}%\n'
                        .format(val_loss, val_acc*100), flush=True)  
                # plot 
                TrainingEval.plot_metrics(acc_list, loss_list,
                            valid_acc_list, valid_loss_list, epoch)


        
        # Save the model every 2 epochs
        if train_idx % 5 == 0:
            # save nn model as checkpoint to restart from
            utils.save_checkpoint(model, optimizer, \
                                  epoch, train_idx,  \
                                  loss_list, acc_list,\
                                  working_dir)
            
            # save nn model as final (weights only) 
#             utils.save_final_model(model, working_dir)
            # log current training status to log file
            LogFile.log_saved_model(steps=nr_of_batches)
            LogFile.log_performance(\
                    acc, loss.item(), ds_type='Training')
            
            # test nn model on test data set 
            if args.testing: 
              
                # get performance of current nn model on test data
                test_loss, test_acc, test_acc_pad, conf_matrix, N_term,C_term, N_pad = \
                            Test.get_performance(
                            model, criterion, \
                            confusion_matrix = True, \
                            pos_acc=True, \
                            debug = args.debug)
                
                # save test set metrics of nn model
                Test.save(test_acc, test_loss, test_acc_pad,  epoch=epoch,
                        step=nr_of_batches)
                
                # plots different model analyses
                Test.plot_confusion_matrix(conf_matrix)
                Test.save_conf_matrix(conf_matrix)
                
                # plot performance prediction on each aa type
                Test.plot_per_class(conf_matrix)
               
                # plot positional accuracy, i.e. how well predicts from N-term and C-term
                Test.plot_pos_acc(N_term, C_term, N_pad)
                
                # log test metrics in log file
                LogFile.log_performance(test_acc, test_loss, ds_type='Test')

        
                

if __name__=="__main__":
    main()
