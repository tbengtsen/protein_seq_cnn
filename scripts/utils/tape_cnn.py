"""
Utils for using Nicki's pytorchtape with Tone's CNN models. 
"""

import os 
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import json
from datetime import date
from datetime import datetime
import shutil
import importlib
from scipy.stats import spearmanr
from scipy.stats import pearsonr
# import utils
# pytorchtape 
# from .pytorchtape.data_utils.vocabs import UNIREP_TONE_VOCAB
from .pytorchtape.data_utils.vocabs import PFAM_2_TONE as pfam_2_tone
from utils import load_checkpoint
##############################################################################

class TapeMetrics():
    '''deals with  metrics/performance, MSE,MAE,S_Corr(spearmanr), of the TAPE task
    '''
    def __init__(self,batch_size,epochs, len_test):
            
        # metrics from batches    
        self.metrics = {'train':{'MSE':[],'MAE':[],'S_Corr':[]},
                        'test':{'MSE':[],'MAE':[],'S_Corr':[]},
                        'val':{'MSE':[],'MAE':[],'S_Corr':[]}}
        self.batch_size = batch_size
        self.epochs = epochs
        self.len_test = len_test
        # keep track of training batches  
        self.steps = {'train':0,'test':0,'val':0}
        self.predictions = np.zeros(len_test)
        self.targets = np.zeros(len_test)
        

    def add_metrics(self, data_type, batch_metrics):
        ''' adds metrics of batch to dictionary of metrics'''
        #check correct input
        assert data_type in ['train', 'test', 'val'], \
            f"Obs wrong data_type in metrics: {data_type}, \
            must be 'train', 'test' or 'val'"
        # add metrics
        self.metrics [data_type]['MSE'].append(batch_metrics['MSE'])
        self.metrics [data_type]['MAE'].append(batch_metrics['MAE'])
        self.metrics [data_type]['S_Corr'].append(batch_metrics['S_Corr'])

    def add_prediction_and_target(self,pred,target,len_batch):
        ''' add prediction, target for each sequence in batch,
        used for plotting later.
        '''
        # get predictions/targets for plotting later
        idx_start = self.steps['test']-len_batch
        idx_end  = self.steps['test']
        self.predictions[idx_start : idx_end] = pred
        self.targets[idx_start : idx_end] = target

    def get_correlations(self,data_type = 'test'):
        '''calculates spearman (S_Corr) and pearson (P_Corr) correlations
        '''
        S_Corr, _ = spearmanr(self.predictions, self.targets)
        P_Corr, _ = pearsonr(self.predictions, self.targets)
        
        return S_Corr, P_Corr
        
        
    def get_avg_metrics(self,data_type = 'test'):
        '''returns the average of the metrics lists '''
        sum_MSE = sum(self.metrics[data_type]['MSE'])
        sum_MAE = sum(self.metrics[data_type]['MAE'])
#         sum_S_Corr = sum(self.metrics[data_type]['S_Corr'])
        MSE = sum_MSE / len(self.metrics[data_type]['MSE'])
        MAE = sum_MAE / len(self.metrics[data_type]['MAE'])
        
        return MSE,MAE
           

        

class TapeOutputs():
    '''Initialise output directory and handlee all output files
    '''
    
    @classmethod
    def set_out_dir(cls, parser_args):
        '''Define and create output directory. 
        If not specified in parser arguments, use same directory
        as the trained CNN model'''
       
        args = parser_args

        # if debugging mode, use debug directory
        if args.debug:
            top_dir = os.path.realpath(__file__).split('scripts')[0]
            out_dir = top_dir+f'models/debugging/tape/{args.task}/'
            # create tape sub-dir
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            
           
        # get name of output dir from nn_model path
        elif args.out_dir is None: 
            model_dir = args.model_weights.strip().split("/")[:-1]
            model_dir = "/".join(model_dir)+"/"
            
            # check if output dir exists/works
            assert os.path.isdir(model_dir), f'Using trained model dir {model_dir} as output dir failed'
            
            # name Tape sub-dir inside trained model dir
            top_dir = model_dir + '/tape/'
            out_dir = top_dir + f'{args.task}/'
            
            # create tape sub-dir
            if not os.path.isdir(top_dir):
                os.makedirs(top_dir)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            
            elif os.path.isdir(out_dir):
                print(f'Obs {out_dir} already exist, some files might be overwritten',\
                      flush=True)
                
           
        # if path specified
        elif args.out_dir is str:
            
            # check path exist
            if os.path.isdir(args.out_dir):
                out_dir = args.out_dir
            elif not os.path.isdir(args.out_dir): 
                # test that path is in out_dir name as well
                assert '/' in args.out_dir[:-1], \
                    'Must specify wanted path to output directory,\
                     not just name:{}'.format(args.out_dir)
                print (f'creating output directory at {args.out_dir}')
                os.makedirs(args.out_dir)
                out_dir = args.out_dir
       

        return out_dir
    
    @classmethod
    def create_log_file(cls, out_dir, parser_args ):
        '''creates log file ''' 
        logfile = out_dir + "logfile_params.log"
        
        with open(logfile, "a+") as f:
            f.write('\n\nINPUT PARAMS:\n======================\n')
            for arg in vars(parser_args):
                f.write('{} :\t\t {}\n'.format(arg, getattr(parser_args, arg)))
            
        return logfile
    
               
    @classmethod
    def log_performance(cls, logfile, MSE,MAE, S_Corr,P_Corr):
        with open(logfile, 'a+') as f:
            f.write('\nPerformance:\n')
            f.write(f'Test MSE :\t {MSE}\n')
            f.write(f'Test MAE :\t {MAE}\n')
            f.write(f'Test S_Corr :\t {S_Corr}\n')

        
     
    ##################
    @classmethod
    def save_metrics(cls, metrics, steps, out_dir):
        '''save the metrics to output directory'''
        # add steps to metrics before saving
        metrics['train']['steps'] = steps['train']
        metrics['test']['steps'] = steps['test']
        metrics['val']['steps'] = steps['val']
        
        out_file = out_dir +'metrics.json'
        with open(out_file, 'w') as json_file:
                json.dump(metrics, json_file)
    
    @classmethod
    def save_predictions(cls,predictions, targets,out_dir):
        '''save the predictions to output directory'''
        dict_pred = {'predictions':list(predictions),'targets':list(targets)}
        out_file = out_dir +'predictions_and_targets.json'
        with open(out_file, 'w') as json_file:
                json.dump(dict_pred, json_file)
                
        
        

    @classmethod
    def plot_predictions(cls, predictions, targets, out_dir, task,\
                         S_Corr=None, P_Corr=None, MAE=None):
        '''Save plot of test predictions vs targets '''
        # plot N-term accuracy 
        fig = plt.figure(figsize=[7,7])
        try:
            fig.suptitle(f'{task}: MAE: {MAE:.2f}, Spearman: {S_Corr:.2f}, Pearson: {P_Corr:.2f}', fontsize=17)
        except: 
            pass
        ax = fig.add_subplot(111) 
        ax.scatter(predictions, targets, marker='.', color='blue')
        ax.legend(loc='upper right', fontsize=15)
        ax.set_xlabel('Predicted ', fontsize=17)
        ax.set_ylabel(r'Experimental', fontsize=17)
        ax.set_title(f'{task}',fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        ax.axhline(0, linewidth=2, color = 'k') 
        ax.axvline(0, linewidth=2, color = 'k')
        max_ = max([ax.get_xlim()[1],ax.get_ylim()[1]])
        min_ = min([ax.get_xlim()[0],ax.get_ylim()[0]])
        ax.set_xlim(min_,max_)
        ax.set_ylim(min_,max_)
        ax.plot([min_,max_], [min_,max_], ls="--", c="0.3",linewidth=3)
        save_path = out_dir+f'/plot_{task}.png'
        plt.savefig(save_path)
        plt.close('all')
        
        # plot histograms 
        # target histogram
        fig = plt.figure(figsize=[12,6])
        fig.suptitle(f'{task}', fontsize=17)
        ax = fig.add_subplot(121) 
        ax.hist(targets,bins=15)
        ax.set_title('Experimental values', fontsize=17)
        ax.set_ylabel(r'freq', fontsize=14)
        ax.set_xlabel(f'{task}', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        if task == 'fluorescence':
            ax.set_xlim(0,4.2)
            ylim = ax.get_ylim() # used below
            ax.set_ylim(0,ylim[1])
            ax.axvline(3,0,ylim[1],color='grey', ls='--')
            ax.text(1,ylim[1]*0.8 , 'Dark', fontsize=14)
            ax.text(3.3,ylim[1]*0.8 , 'Bright', fontsize=14)
            ax.set_xlabel(r'log fluorescence', fontsize=14)


        
        
        # predictions histogram
        ax = fig.add_subplot(122) 
        ax.hist(predictions,bins=15)
        ax.set_title('Predicted values', fontsize=17)
        ax.set_xlabel(f'{task}', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        if task == 'fluorescence':
            ax.set_xlim(0,4.2)
            ax.set_ylim(0,ylim[1])
            ax.axvline(3,0,ylim[1],color='grey', ls='--')
            ax.text(1,ylim[1]*0.8 , 'Dark', fontsize=14)
            ax.text(3.3,ylim[1]*0.8 , 'Bright', fontsize=14)
            ax.set_xlabel(r'log fluorescence', fontsize=14)


        # save
        save_path_hist = out_dir+f'/plot_histogr_{task}.png'
        plt.savefig(save_path_hist)
        plt.close('all')
    
        
        return save_path
    
    
    @classmethod
    def plot_downstream_training(cls, task, out_dir, metrics ,MAE=None, S_Corr=None, P_Corr=None, epochs=1,lr=0):
        fig = plt.figure(figsize=(10,7))
        title = f'{task} - {epochs} epochs - lr={lr},  test metrics: \n \
            MAE {MAE:.2f}, Spearman: {S_Corr:.2f}, Pearson: {P_Corr:.2f}'
        fig.suptitle(title, fontsize=17)
        
        # train
        ax = fig.add_subplot(231)
        ax.set_title('Loss MSE - Training')
        ax.plot(metrics['train']['MSE'],color='blue')
        ax.set_label('batches')
#         if epochs>1:
#             epoch_steps = int(len(metrics['train']['MSE'])/epochs)
#             for i in range(epochs):
#                 x = epoch_steps*(i+1)
#                 ax.axvline(x, ax.get_ylim()[0],ax.get_ylim()[1], ls='--')
#             ax.set_label('batches (colours=epocs)')   
        
        ax = fig.add_subplot(234)
        ax.set_title('Spearman - Training batches')
        ax.plot(metrics['train']['S_Corr'])
        ax.set_label('batches')
        
        # validation
        ax = fig.add_subplot(232)
        ax.set_title('MSE - Valid')
        ax.plot(metrics['val']['MSE'][:],color='blue')
            
        ax = fig.add_subplot(235)
        ax.set_title('Spearman - Valid')
        ax.set_label('batches')
        ax.plot(metrics['val']['S_Corr'],color='blue')
                
        # test
        ax = fig.add_subplot(233)
        ax.set_title('MSE - Test')
        ax.plot(metrics['test']['MSE'])
        avg_mse = sum(metrics['test']['MSE'])/len(metrics['test']['MSE'])
        ax.axhline(avg_mse, color='red')
        ax.set_label('batches')

        ax = fig.add_subplot(236)
        ax.set_title('Spearman - Test')
        ax.plot(metrics['test']['S_Corr'])
        ax.axhline(S_Corr, color='red')
        ax.set_label('batches')

        save_path = out_dir+f'/plot_downstream_training_{task}.png'
        plt.savefig(save_path)
        plt.close('all')

        
        

class PerformJSON():
    '''returns JSON dictionary from downstream_performance.json'''
    def __init__(self):
        
        # json file path 
        self.cwd = os.path.dirname(os.path.abspath(__file__)) 
        self.folder = self.cwd + '/../../models/downstream_performance/'
        self.bckup_folder = self.cwd + '/../../models/downstream_performance/bck_up/'
        self.json = self.folder
        +'downstream_performance_upst_params_not_fixed.json' # XX changed
        self.perform_dict = self.get_perform_dict()
        
    def get_perform_dict(self):
        ''' returns the opened json file dictionary .
        '''
        # checks if path correct and
        assert os.path.isdir(self.folder), f'directory {self.folder} \
                                        with json file does not exists'
        assert os.path.isfile(self.json), f'json performance\
                                        file:{self.json} does not exists'
        # opens json file 
        with open(self.json,'r') as json_file:
            perform_dict = json.load(json_file)

        return perform_dict
    
    def save_perform_dict(self, perform_dict,debug=False):
        ''' saves  dict to  json file dictionary .
        '''
        # debug 
        if debug:
            self.debug_version(perform_dict)
        else:
            # save old version 
            self.bckup()
            # opens json file 
            with open(self.json, 'w') as json_file:
                json.dump(perform_dict, json_file)

    def bckup(self):
        # save old version 
        now =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        json_bckup = self.bckup_folder + f'downstream_performance_{now}.json'
        shutil.copyfile(self.json, json_bckup)

    def debug_version(self,perform_dict):
        print('debug version')
        debug_json = ''.join(self.json.split('.json')[:-1]) + '_debug.json'
        with open(debug_json, 'w') as json_file:
            json.dump(perform_dict, json_file)
        print(f'debug version saved in {debug_json}')

        


         
        
class SavePerform2json(PerformJSON):
    '''add mdoels downstream performance to  performance table:\
    downstream_performance.json.
    Args:
        nn_model: name of nn model architecture (str)
        model_params: dict of model_params
        
    '''
    def __init__(self, nn_model,  model_params, avg_metrics, tape_task=None, dms_task=None, \
                  plots=None, paths = None, \
                 notes=None,debug=False):
        
        super().__init__()
    
        if debug:
            self.nn_model = self._rename_if_debug(nn_model)
        else:
            self.nn_model =self._add_model(nn_model, model_params)
                    
        self.model_params =  model_params
        self.MSE = round(avg_metrics['MSE'],3)
        self.MAE = round(avg_metrics['MAE'],3)
        self.S_Corr = round(avg_metrics['S_Corr'],3)
        self.tape_task= tape_task
        self.dms_task= dms_task
        self.plots = plots
        self.paths = paths
        self.notes = notes
        self.date = str(date.today())

        # add inputs to table/JSON 
        self._add_performance()
        self._add_notes()
        self._add_date()
        self._add_plots()
        self._add_paths()
        self.save_perform_dict(self.perform_dict, debug)
                
        
    def _rename_if_debug(self,nn_model):
        '''change name of input to table to DEBUG if debug mode '''
        nn_model = 'DEBUG'

        return nn_model


    def _add_model(self, nn_model,model_params):
        '''create new input lline to dictionary based on model''' 
        exists, nn_model = self._is_model_in_dict(nn_model, model_params)
        
        if not exists:
            self.perform_dict[nn_model] = {}
            self.perform_dict[nn_model]['MODEL_PARAMS']= model_params
            self.perform_dict[nn_model]['TAPE'] = {}
            self.perform_dict[nn_model]['DMS'] = {}
            self.perform_dict[nn_model]['PATHS'] = {}
            self.perform_dict[nn_model]['PLOTS'] = {}
            self.perform_dict[nn_model]['NOTES'] = 'Additional notes\n'
            self.perform_dict[nn_model]['DATE_LAST_ADDED'] = ''
        
        return nn_model

    def _add_performance(self):
        # check task exists in dir 
        Assertions.tape_task(self.tape_task)
        if self.tape_task != None:
            self.perform_dict[self.nn_model]['TAPE'][self.tape_task] = {}
            self.perform_dict[self.nn_model]['TAPE'][self.tape_task]['MSE'] = self.MSE
            self.perform_dict[self.nn_model]['TAPE'][self.tape_task]['MAE'] = self.MAE
            self.perform_dict[self.nn_model]['TAPE'][self.tape_task]['S_Corr'] = self.S_Corr
        if self.dms_task != None:
            Assertions.dms_task(self.dms_task)
            self.perform_dict[self.nn_model]['DMS'][self.dms_task] = {}
            self.perform_dict[self.nn_model]['DMS'][self.dms_task]['MSE'] = self.MSE
            self.perform_dict[self.nn_model]['DMS'][self.dms_task]['MAE'] = self.MAE
            self.perform_dict[self.nn_model]['DMS'][self.dms_task]['S_Corr'] = self.S_Corr

    def _add_notes(self):
        if self.notes is not None:
            notes = 'Additional notes\n' + f'{self.notes}'
            self.perform_dict[self.nn_model]['NOTES'] = notes


    def _add_date(self):
        self.perform_dict[self.nn_model]['DATE_LAST_ADDED'] = self.date


    def _add_plots(self):
        if self.plots is not None:
            Assertions.link_2_plots(self.plots)
            for plot, path in self.plots.items():
                 self.perform_dict[self.nn_model]['PLOTS'][plot] = path

    def _add_paths(self):
        if self.paths is not None:
            Assertions.paths(self.paths)
            for task, path in self.paths.items():
                 self.perform_dict[self.nn_model]['PATHS'][task] = path


    def _is_model_in_dict(self,nn_model,model_params):
        '''check if model name in table, if yes, check if same model \
        params (kernel size etc), if not, rename self.nn_model to\
        distinguish between same model archit but different model params
        '''
        # check if model name in dict 
        if nn_model in  self.perform_dict.keys() : 
            
            # check if model params in dict
            dict_model_params = self.perform_dict[nn_model]['MODEL_PARAMS']
            
            if model_params == dict_model_params:
                print('Model already in table, might overwrite some data')
                return True, nn_model
            
            # rename model 
            elif model_params != dict_model_params:
                # give new name to model archit that includes kernelsize
                nn_model_renamed = nn_model + f"_ks{model_params['ks_conv']}"
                print(f'model architecture already in tabel, \
                        but not the same params, renaming model to \
                        {nn_model}',flush=True)
                            # check if model params in dict
                if nn_model_renamed in self.perform_dict.keys():
                    print('Model already in table, might overwrite some data')
                    return True, nn_model_renamed
                    print('Model already in table, might overwrite some data')
                else:
                    return False, nn_model_renamed 
        
        elif nn_model not in self.perform_dict.keys() :
            return False, nn_model

        
                
class Assertions():
    
    @classmethod 
    def tape_task(cls, tape_task):
        '''check all is correct'''

        assert tape_task in ["fluorescence", \
                                "stability", \
                                "proteinnet",\
                                "remotehomology",\
                                "secondarystructure",\
                                "pfam",\
                                None], f'Invalid Tape task: {tape_task}'
    @classmethod
    def dms_task(cls, dms_task):
        assert self.dms_task in ['proteinG', \
                                'Fowler_1D5R', \
                                'Fowler_2H11',\
                                None], f'invalid dms_task {dms_task}'
        
    @classmethod
    def model_params(cls, input_model_params, table_model_params):
        assert input_model_params.keys() == table_model_params['MODEL_NAME'].keys(),\
            f"model params not correctly added, this is important, please use\
            format of keys: {dict_perform['MODEL_NAME'].key()}"
        
    @classmethod
    def link_2_plots(cls,dict_plots):
        if dict_plots is not None:
            # check type for plot
            for task in dict_plots.keys():
                assert task in ["fluorescence", \
                                    "stability", \
                                    "proteinnet",\
                                    "remotehomology",\
                                    "secondarystructure",\
                                    "pfam"], f'Invalid plot task: type {task}'
            # check path exists   
            for path in dict_plots.values():
                assert os.path.isfile(path), f'File does not exist : {path}'

            
    @classmethod
    def paths(cls, dict_paths):
        if dict_paths is not None:
            # check type of task
            for dir_type in dict_paths.keys():
                    assert dir_type in ['model_dir','tape_dir', 'DMS_dir', 'log_file'], f'Invalid directory type for path {dir_type}'
            # check path exists 
            for path in dict_paths.values():
                if path is not 'logfile_params.log':
                    assert os.path.exists(path), f'File does not exist : {path}'        
        
        
    

class RepresentationModel():
    '''Initialise the upstream trained representation model e.g. 
    outputdirectory, loading the nn achitechture, loading the
    CNN trained model etc.
    ''' 
    
    @classmethod
    def import_nn_archit(cls, nn_model):
        '''loads pytorch CNN model architecture (nn_model)from path'''

        try:
            path = './nn_models/' + nn_model
            spec = importlib.util.spec_from_file_location('nn_module', path)
            nn_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nn_module) 

        except FileNotFoundError:
            print('NN module {} not found in path {}.\
            Specify name of module only nnd not path'.format(nn_model,path))
            sys.exit('nn module not found.')


        return nn_module.ConvNet
    
    @classmethod
    def is_params_identical(cls, parser_args):
        '''Check if the parsed model params are identical to the ones
        used for training the model 
        '''
        args = parser_args

        params = {} # list of train vs parsed params for output
        identical = True
        
        # read training model params from log file: 
        out_dir = args.model_weights.strip().split("/")[:-1]
        logfile = "/".join(out_dir)+ "/logfile_params.log"
        
        with open(logfile, 'r') as f:
            for line in f.readlines():
                # assert if training params and parsed params is identical
                if 'kernel_size' in line: 
                    # get training param 
                    train_ks = int(line.split(':')[-1])
                    # add params to dictionary for output to slurm
                    params['kernel_size'] = (train_ks, args.kernel_size)
                    # set return to False if not identical 
                    identical = False if train_ks != args.kernel_size else True
                
                elif 'stride :' in line: 
                    train_str = int(line.split(':')[-1])
                    params['stride'] = (train_str, args.stride)
                    identical = False if train_str != args.stride else True
                    

                elif 'padding :' in line: 
                    train_pad = int(line.split(':')[-1])
                    params['padding'] = (args.padding, train_pad)
                    identical = False if train_pad != args.padding else True
                    
                
                elif 'ks_pool :' in line: 
                    ks_pool = int(line.split(':')[-1])
                    params['ks_pool'] = (args.ks_pool, ks_pool)
                    if ks_pool is not None and ks_pool != -1:
                        identical = False if ks_pool != args.ks_pool else True
                
                elif 'str_pool :' in line: 
                    str_pool = int(line.split(':')[-1])
                    params['str_pool'] = (args.str_pool, str_pool)

                    if str_pool is not None and str_pool != -1:
                        identical = False if  str_pool != args.str_pool else True
                
                elif 'pad_pool :' in line: 
                    pad_pool = int(line.split(':')[-1])
                    params['pad_pool'] = (args.pad_pool, pad_pool)
                    if pad_pool is not None and pad_pool != -1:
                        identical = False if pad_pool != args.pad_pool else True
               
        return identical, params


    
    @classmethod
    def load_model_weights (cls, trained_model_path, model):
        '''Load trained model, using the checkpoint file and 
        loaded  model architecture from nn_model 
        '''

        # define optimizer as needed to load chckpt-file, not used 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        # load trained model weights into init model architectyre from file 
        model, optimizer, epoch_start, train_ds, loss_list, acc_list = \
            load_checkpoint(model, optimizer, filename=trained_model_path) 

            
        return model

    

class Tape2CNN():
    ''' Functions to convert TAPE's seq batches (Nicki's pytorchTape) to be
    used with pretrained CNN models.  
    '''  
        
    def encode_batch(self,batch):
        # convert encoding to fit with CNNs, in batch[primary] = sequences 
        seq_converted = self.convert_aa_encoding(batch['primary']) 
        seq_one_hot = self.one_hot(seq_converted)
        # transpose to input seq as vector
        seq_transposed = torch.transpose(seq_one_hot,1,2)
        # add to data 
        encoded_batch = batch
        
        encoded_batch['primary'] = seq_transposed
        # convert to cuda 
        if torch.cuda.is_available():
                encoded_batch.cuda()
        
        return encoded_batch
        
    def convert_aa_encoding(self, batch):
        '''convert input data (pfam vocab) to aa coding used by the CNN 
        '''
        #to numpy to allow conversion
        batch = batch.cpu().numpy() 

        # convert encoding
        batch = np.array([[pfam_2_tone[aa] for aa in seq] for seq in batch])

        # check if any seq contains [X,Z,J,B] and discard those seq in batch
        batch = self.check_ambigiuos_aa(batch)

        # convert back to tensor
        batch = torch.tensor(batch,dtype=torch.long)

        return batch
  
    def check_ambigiuos_aa(self, batch):
        ''' Delete rows with np.nan in converted aa encoding
        This is done as the CNN trained models did not include ambigous aa
        and these are converted to np.nan
        '''
        batch = batch[~np.isnan(batch).any(axis=1)]
        return batch
     
    def one_hot(self, batch):
        '''Convert to one hot encoding'''
        num_classes=25
        if torch.cuda.is_available():
            to_one_hot = torch.eye(num_classes, dtype=torch.float32, device='cuda')
        else:
            to_one_hot = torch.eye(num_classes,dtype=torch.float32)    
    
        batch = to_one_hot[batch]
        return batch
