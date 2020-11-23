import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import os 
# ensures tenflow does not use non-allocated GPUs, 
# as tensorflow imported in utils.pytorchtape
try:
    gpu_nr = (os.environ['CUDA_VISIBLE_DEVICES'])
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_nr
except: 
    pass

# pytorchtape 
from utils.pytorchtape import get_task #, vocab
# tape on CNN
from utils import TapeMetrics
from utils import Tape2CNN
from utils import RepresentationModel
from utils import TapeOutputs
from utils import SavePerform2json




############################################################
##  Input arguments   
############################################################



# input arguments 
parser = argparse.ArgumentParser(description = 
        '\n \n \
         Evaluate pretrained CNN representation models on TAPE tasks. \n \
         ====================================================\n ', 
         formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument('--architecture', dest = "nn_model", required=True, 
            help='Name of upstream nn architecture model (class) . Must be defined in scripts/nn_models/\
            and imported in __init__.py\ in that directory.')

parser.add_argument('--trained_model',dest='model_weights',
        required=True, default = None,         
        help='path to pt model file to analyse. Must be saved as in utils.py')

## nn model params ##
parser.add_argument('--kernel_size', dest = 'kernel_size', 
        default=None, required = False, type=int,
	    help='Window size of CNN pretrained model')

parser.add_argument('--stride', dest='stride',
        default=None, required=False, type=int,
        help ='Stride size for moving kernels/windows of CNN pretrained \
        model')

parser.add_argument ('--padding',dest='padding',
        default=None,required=False,  type=int,
	    help='padding on each sequence after each convolution of CNN \
        pretrained model.\n rule: -kernel+2P+1) = 0. E.g -5+2*2+1 =1 \
        for k_size=5')


parser.add_argument('--ks_pool', dest = 'ks_pool', 
        default=None, required = False, type=int,
	    help='Window size for avg pooling of CNN pretrained model')

parser.add_argument('--str_pool', dest='str_pool',
        default=None, required=False, type=int,
        help ='Stride size for moving kernels/windows in pooling of CNN \
        pretrained model')

parser.add_argument ('--pad_pool',dest='pad_pool',
        default=None,required=False,  type=int,
	    help='padding on each sequence after each avg pool of CNN pretrained\
        model.\n rule: -kernel+2P+1) = 0. E.g -5+2*2+1 =1 for k_size=5')
	    

parser.add_argument('--lr','-lr',dest='lr',
        default = 0.001, required=False,  type=float,
	    help='Learning rate for downstream gradient. Recommended based on \
        preliminary test is lr_{stability}=<0.001 and lr_{flourence}=<0.001. \
        Obs! this is an important parameter for this case, so do test it. ')

parser.add_argument('--epochs',dest='epochs', 
        default=5,  type=int, required=False,
	    help='Number of epochs for downstream training,\
              i.e. iterations though training set')

parser.add_argument ('--batch_size', dest='batch_size',
        default=100, required=False, type=int,
        help='Batch size for downstream tape task training. Recommended based\
        on preliminary tests is 100. Obs 10 is to low for learning. ' )

## Tape settings ##
parser.add_argument('--tape_task',dest='task',
        required=True, default = "stability", #nargs='+', type=str, does not work with multiple tasks and choices
        choices={"fluorescence","proteinnet","remotehomology", \
                 "secondarystructure", "stability", "pfam"},        
        help='What TAPE task/dataset to downstream predict on, choose from:\
             "fluorescence","proteinnet","remotehomology",\
             "secondarystructure", "stability" or "pfam" ')

# output dir 
parser.add_argument('--out_dir',dest='out_dir',
        required=False, default = None,         
        help='Where to store output data and analyses. \
        Default directory is set to the directory where the trained model (weights) is located ')

parser.add_argument('--debug',dest='debug', action="store_true", 
	    help='whether to only debug script')

                 


# -----------------------------------------------------------
# ------- TRAINING DOWNSTREAM MODEL ------
# -----------------------------------------------------------


def main():
    
    # get input arguments 
    args = parser.parse_args()
    print (args.nn_model) 
    # define output directory 
    out_dir = TapeOutputs.set_out_dir(args)
    print(f'\n\n -- Output dir: \n\t{out_dir}',flush=True)
    
    # log parsed options
    logfile = TapeOutputs().create_log_file(out_dir, args)
    
    
    ## GET UPSTREAM TRAINED MODEL  ##
    # import representation nn model architecture
    ConvNet = RepresentationModel.import_nn_archit(args.nn_model)
    
    # check if parsed model params are identical to the training params
#     identical, params = RepresentationModel.is_params_identical(args)
#     if identical is False: 
#         sys.exit(f'The parsed model params is not identical to the once used \
#         for training. Existing.\nPrinted below are the parsed param vs \
#         training param:\n {params}')
    
    # Init nn model  with parsed model params
    model = ConvNet(args.kernel_size, args.stride, args.padding,\
                        args.ks_pool, args.str_pool, args.pad_pool)


    # load trained model weights into architecture
    model = RepresentationModel.load_model_weights(args.model_weights, model)
    
    print (f'\n ==> Loaded CNN trained model weights from:\n\t {args.model_weights}', \
           flush=True) 
    
    # keep model in eval mode
    model=model.eval()
    
    
    ## GET TAPE TASK DATA AND DOWNSTREAM MODEL ##
    
    # Define tape task to evaluate on, e.g stability, secondary structure
    task = get_task(args.task)(model, fix_embedding=False) # XX changed 
    if torch.cuda.is_available(): 
            task = task.cuda()   
    print(f' \n -- Evaluating on TAPE task(s):\n\t{args.task}  ',flush=True)
    
    # get split TAPE dataset
    train, val, test = task.get_data(batch_size=args.batch_size, \
                                     max_length=498)
    
    # get downstream optimizer
    optimizer = torch.optim.Adam(task.parameters(), args.lr) #default 0.001
    

    # Init performances metrics
    metrics = TapeMetrics(args.batch_size, args.epochs, len(test))
   



        
    ## TRAINING DOWNSTREAM MODEL ON TAPE TASK ## 
    
    for epoch in range(args.epochs):
        for batch in train:

            metrics.steps['train'] += len(batch['primary'])
            
            #if in debug mode, loop over fewer datapoints 
            if args.debug and metrics.steps['train'] > 1000: break


            # convert tape encoding to fit with CNN model (using vocab) 
            # and onehot encode
            batch = Tape2CNN().encode_batch(batch)
            # calculate loss og metrics of downstream model
            # each batchs contains both sequence and downstream Tape task target. 
            loss, batch_metrics = task.loss_func(batch)
            metrics.add_metrics('train', batch_metrics)


            ## switch model to training mode, clear gradient accumulators ##
            task.train()
            optimizer.zero_grad()

            ##  Backprop and perform Adam optimisation  ##
            loss.backward()
            optimizer.step()
        
        # validation each epoch
        task.eval()
        with torch.no_grad():
            for batch in val:
                metrics.steps['val'] += len(batch['primary'])
                # debugging mode
                if args.debug and metrics.steps['val'] > 500: break
                # convert to CNN input
                batch = Tape2CNN().encode_batch(batch)
                #get batch performance
                _ , batch_metrics = task.loss_func(batch)
                # save batch performance
                metrics.add_metrics('val', batch_metrics)



    # evaluate on test set: 
    task.eval()
    with torch.no_grad():
        for batch in test:
            metrics.steps['test'] += len(batch['primary'])
            # debugging mode
            if args.debug and metrics.steps['test'] > 500: break
            
            # convert tape encoding to fit with CNN model (using vocab) 
            # and onehot encode
            batch = Tape2CNN().encode_batch(batch)
            
            # get predictions/targets for plotting later
            pred, target = task.get_prediction(batch)
            metrics.add_prediction_and_target(pred, target, \
                                              len(batch['primary']))
#             # get metrics
            _ , batch_metrics = task.loss_func(batch)
            metrics.add_metrics('test', batch_metrics)



    
    # get test performance (average)
    MSE, MAE = metrics.get_avg_metrics('test')
    S_Corr, P_Corr = metrics.get_correlations()
    
    # log performance
    TapeOutputs.log_performance(logfile, MSE,MAE, S_Corr,P_Corr)
    
    # save  metrics in output dir #
    TapeOutputs.save_metrics(metrics.metrics, metrics.steps, out_dir)
    TapeOutputs.save_predictions(metrics.predictions, \
                                 metrics.targets,\
                                 out_dir)
#     # plot training,val,test data
    TapeOutputs.plot_downstream_training(args.task, out_dir, \
                metrics.metrics, MAE=MAE,S_Corr=S_Corr, P_Corr=P_Corr, \
                epochs=args.epochs,lr=args.lr)
    
    # plot predictions vs targetes  and get path of plot
    plot_path = TapeOutputs.plot_predictions(metrics.predictions,\
                                             metrics.targets, \
                                             out_dir,task=args.task,
                                             S_Corr=S_Corr, P_Corr=P_Corr, \
                                             MAE=MAE )
   
    ## save to json dict that contains all model performance ##
    # easy input of model params to json
    model_params = {'latent_size':model.latent_size,'layers':model.layers,\
                    'fully_con':model.fully_con, 'compression_seq':model.seq,\
                    'channels':model.chnls, 'ks_conv':args.kernel_size,\
                    'str_conv':args.stride,'pad_conv':args.padding,\
                    'ks_pool':args.ks_pool, 'str_pool':args.str_pool,\
                    'pad_pool':args.pad_pool}
    paths = {'tape_dir':out_dir, 'log_file':logfile}
    plots = {f'{args.task}':plot_path}
    avg_metrics = { 'MSE':MSE, 'MAE':MAE, 'S_Corr':S_Corr}
                                              
    # save to json 
    SavePerform2json(args.nn_model, model_params,avg_metrics, \
                        tape_task=args.task, plots=plots, paths = paths, \
                        notes=None,debug=args.debug)
        
       
                 
                 


    



if __name__=="__main__":
    main()
