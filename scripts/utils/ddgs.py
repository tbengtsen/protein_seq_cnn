'''
Utils for predicting ddgs on Mahers cleaned ddg dataset,
using the representation from traineed CNNs. 
'''
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr

from IPython.display import display, Markdown
import nglview as nv
import MDAnalysis as mda

import utils
from utils import to_one_hot
from utils import get_triAA_to_int
from utils.tape_cnn import PerformJSON


##############################################################################
class DMSOutputs():
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
            sub_dir = top_dir+'models/debugging/dms/{}/'.format(args.task)
            # create tape sub-dir
            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)
            
           
        # get name of output dir from nn_model path
        elif args.out_dir is None: 
            top_dir = args.model_weights.strip().split("/")[:-1]
            top_dir = "/".join(top_dir)+"/"
            # name dms sub-dir inside trained model dir
            sub_dir += top_dir + '/DMS'
            out_dir = sub_dir + f'/{args.protein}/'
            
            # create the directories
            # check if output dir exists/works
            message = f"Using trained model dir {top_dir} as output  failed"
            assert os.path.isdir(top_dir), message
            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            
            elif os.path.isdir(out_dir):
                print(f'Obs {out_dir} already exist, some files might be\
                overwritten', flush=True)
                
            # log to slurm output
            print ('Obs! No output directory given. Using same directory \
                    as trained model as output dir: \{}'.format(out_dir),\
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
    def plot_predictions(cls, rfr, rf_test_set, idx_cor, idx_incor, \
                         MAE, S_Corr, out_dir, protein, best_params):
        '''outputs plot of RF predictions '''
        if protein is 'protein_g':
            y_label = r"Experimental ∆∆G [kcal] $\sigma$=0.1"
            x_label = "Predicted ∆∆G"
            ylim = [-2, 4.1]
            xlim = [-2, 4.1]
        else: 
            y_label = " Experimental DMS "
            x_label = " Predicted DMS "
            ylim = [-0.4,1.3]
            xlim = [-0.4,1.3]
            
        fig, ax = plt.subplots(1, 2, sharey=False, figsize=(8, 16), constrained_layout=True) 
        tst = f'S_Corr: {S_Corr:.2f} \nMAE: {MAE:.2f}'
        n_trees = best_params['n_estimators']

        ax[0,0].scatter(rf_test_set[:,-1][idx_cor], rf_test_set[:,-2][idx_cor], color='blue', label=tst)
        ax[0,0].scatter(rf_test_set[:,-1][idx_incor], rf_test_set[:,-2][idx_incor], color='red', label='AE incorrect pred')
        ax[0,0].legend(loc='upper left', fontsize=15)
        ax[0,0].set_xlabel(x_label, fontsize=17)
        ax[0,0].set_ylabel(y_label, fontsize=17)
        ax[0,0].set_title(f"Trained on \nn_trees:{best_params['n_estimators']}", fontsize=20)
        ax[0,0].tick_params(axis='both', labelsize=16)
        ax[0,0].set_xlim(xlim[0],xlim[1])
        ax[0,0].set_ylim(ylim[0],ylim[1])
        ax[0,0].axhline(0, linewidth=2, color = 'k') 
        ax[0,0].axvline(0, linewidth=2, color = 'k')
        ax[0,0].plot(ax[0,0].get_xlim(), ax[0,0].get_ylim(), ls="--", c=".3")

        # Feature importance
        ax[0,1].bar(np.arange(rfr.n_features_),rfr.feature_importances_)
        ax[0,1].set_xlabel('latent space vector ', fontsize=17)
        ax[0,1].tick_params(axis='both', labelsize=16)
        ax[0,1].set_title('feature importance', fontsize=20)
        save_path = out_dir+f'/plot_{protein}.png'
        plt.savefig(save_path)
        plt.close('all')
        
        return save_path

    
    
class HandleFasta():
    '''handles everything with the ddg/dms files for a given protein'''
        
    def _get_fasta(self,protein):
        '''returns path to fasta filee with protein sequence'''        
        if protein is 'protein_g':
            fasta = '../../data/ddgs/proteingmayo/raw/fasta/1PGA.fasta.txt'
        elif protein is "2H11" :
            fasta = '../../data/ddgs/dms_fowler/raw/fasta/2H11.fasta.txt'
        elif protein is "1D5R":
            fasta = '../../data/ddgs/dms_fowler/raw/fasta/1D5R.fasta.txt'
        return fasta
    

    def _fasta_2_vocab(self,protein, max_l=498):
        '''convert ddg's fasta file to a sequence that the CNN models
        can read (like in preprocess_data.py )
        '''
        fasta = self._get_fasta(protein)
        with open(fasta, 'r') as f:
            seq = ""
            for line in f:
                # add to protein seq
                if not line[0] == '>':
                    seq += line.replace("\n","")

            # convert aa to integers        
            int_seq = self._aa_seq_to_int(seq)
            ## pad seq with 0's ##
            int_seq = self._pad_seq(int_seq, max_l)

        return np.array(int_seq)
    
    def vocab_2_one_hot(self,integer_sequence):
        '''takes preprocessed fasta seq and converts to one hot for input to\
        CNN models
        '''
        seq = torch.tensor(integer_sequence,dtype=torch.long)
        
        if  torch.cuda.is_available(): 
            seq.cuda()
        # convert to one hot 
        seq_OH = utils.to_one_hot(seq)

        # convert for channels to come first
        seq_OH = torch.transpose(seq,0,1) #transpose dim 1,2 => channels=aa

        # add "batch" dimension=1 for model to work
        seq_OH = seq [None, :, :]
        
        return seq_OH 
        
    def _pad_seq(self, seq, max_l):
        '''
        Pads the integer sequence with 0's up to max_l+2 
        '''
        padded_seq = [0]*(max_l+2) # +2 as start and stop added to seq 
        padded_seq[:len(seq)] = seq

        return padded_seq
    
    def _aa_seq_to_int(self,seq):
        """
        Return the sequence converted to integers as a list 
        plus start(23) and stop(24) integers. From Unirep. 
        """
        return [23] + [aa_to_int[aa] for aa in seq] + [24]

    
class HandleProtein(HandleFasta):
    def __init__(self,protein):
        
        self.protein = protein
        self._assert_protein ()
        # from HandleFasta
#         self.seq_labels = self._fasta_2_vocab(protein,max_l=498)
        self.sequence = self._fasta_2_vocab(protein, max_l=498)
        self.sequence_one_hot = self.vocab_2_one_hot(self.sequence)
        
    def _assert_protein(self):
        msg = f"{self.protein} is not a valid option for DMS, "
        msg += "choose between:protein_g, 2H11 or 1D5R. "
        assert self.protein in ["protein_g","2H11","1D5R"], msg
            
            

    
    
class HandleDMS():
    '''returns dms file in pandas format for the given protein. Takes only 
    "protein_g","2H11","1D5R"
    '''
    def __init__(self,protein):
        
        self.protein=protein
        self._assert_protein ()
        self.dms_path = self._get_DMS_path()
        self.dms_file = self._read_DMS_file() # pandas 


    def _assert_protein(self):
        assert self.protein in ["protein_g","2H11","1D5R"], \
            f"{self.protein} is not a valid option for DMS, choose between:\
           protein_g, 2H11 or 1D5R. "
    
    def _get_DMS_path(self):
        '''return path to ddg file with all ddg/DMS measurements'''
        if self.protein is 'protein_g':
            dms_file = '../../data/ddgs/proteingmayo/processed/ddgs/proteingmayo.txt'
        elif self.protein is "2H11" :
            dms_file = '../../data/dms_fowler/processed/ddgs/dms_fowler_2H11.txt'
        elif self.protein is "1D5R" :
            dms_file = '../../data/ddgs/dms_fowler/processed/ddgs/dms_fowler_1D5R.txt'
        return dms_file

    def _read_DMS_file(self):
        '''return pandas matrix of dms file'''
        dms_file = pd.read_csv(self.dms_file, sep="\s+", header = None, 
                        names=['PDB', 'CHAIN', 'WT', 'RES', 'MUT', 'DMS'])
        
        return dms_file
    
    def __len__(self):
        return len(self.dms_file['ddg'])



class RepresentationsOfDMS():
    ''' do not want to make a doc, it is too messy... '''
    def __init__(self,protein, model):
        self.protein = protein
        self.model = model
        self.lat_space = self.model.latent_size
        self.dms_file = HandleDMS(protein).dms_file # pandas reading
        self.wt_seq =  HandleProtein(protein).sequence # integer labels
        self.wt_seq_OH = HandleProtein(protein).sequence_one_hot
        self.wt_embed = self.model.embed(self.wt_seq_OH)
        self.int_matrix, self.matrix_idx = self.init_dms_matrix()
        self.dms_matrix = self.add_dms_2_matrix()

        
    def init_dms_matrix(self):
        '''initialize matrix need for RF where each row  contains:
            WT representation, 
            mutations representations, 
            difference in wt and mut representation
            wt labels (integers from vocab)
            idx of mutation in sequence
            wt aa label (in vocab integer)
            mut aa label (in vocab integer)
            true/false prediction of AE on the position
            targets ddg from experiments 
            ''''
        lat_space = self.latent_space # I am lazy, taken from other code
        # define output  matrix
        matrix_idx = {}
        matrix_idx['idx_wt_rpr'] = lat_space # first :ls1 is wt representation
        matrix_idx['idx_mut_rpr'] = lat_space*2 # ls1:ls2 mut representation
        matrix_idx['idx_diff_rpr'] = lat_space*3 # ls2:ls3 = difference: wt repr - mut rep
        matrix_idx['idx_wt_seq'] = lat_space*3 + 500 # ls2:ls3 is wt labels seq()
        matrix_idx['idx_mut_seq'] = lat_space*4 + 500 # ls2:ls3 is mut labels seq()
        
        matrix_idx['idx_pos'] = matrix_idx['idx_mut_seq'] # col in matrix that defines of mutation in seq
        matrix_idx['idx_wt'] = matrix_idx['idx_mut_seq'] + 1  # wt aa label 
        matrix_idx['idx_mut'] = matrix_idx['idx_mut_seq'] + 2  # mut aa label 
        matrix_idx['idx_AE_pred'] = matrix_idx['idx_mut_seq'] + 3 # column where AE pred are def
        matrix_idx['idx_dms'] = matrix_idx['idx_mut_seq'] + 4 # column where exp ddgs are determine
         # initialise matrix, where rows = each mutation, and columns
        dms_matrix = \
            np.zeros((len(self.dms_file),matrix_idx['idx_mut_seq']+5))
        
        return dms_matrix, matrix_idx

    
    
    def add_dms_2_matrix(self):
        ''' add mutations and wt from dms file to each row in matrix'''
        
        # add wt embedding and wt seq to matrix 
        dms_matrix = self._add_wt_2_matrix(self.int_matrix, self.matrix_idx)
        
        for idx, row in self.dms_file.iterrows():
            
            # get info about mutations 
            dms_exp, idx_mut, wt_aa, mut_aa, mut_seq, mut_embed = \
                            self._get_mutation(row)
            
            # add mutation to matrix 
            dms_matrix = self._add_mut_2_matrix(idx, dms_matrix, self.matrix_idx,\
                                                dms_exp, idx_mut, wt_aa, mut_aa, \
                                                mut_seq, mut_embed )
            
            
        return dms_matrix
            
            
            


    def _add_wt_2_matrix(self, dms_matrix,matrix_idx):
        ''' add wt representation to all rows in matrix'''
        
        ## add wt_repr to first 500 columns in ddgs_repr
        dms_matrix[:,:matrix_idx['idx_wt_rpr']] = self.wt_embed.cpu().numpy()

        ## add WT labels (integers) to first columns between 1000:1500
        dms_matrix[:, matrix_idx['idx_dif_rpr']:matrix_idx['idx_wt_seq']] = self.wt_seq
        
        return dms_matrix
    
    
    
    def _add_mut_2_matrix(self, idx, dms_matrix, matrix_idx,\
                          dms_exp, idx_mut, wt_aa, mut_aa, \
                          mut_seq, mut_embed  )
        
        ## add mut embedding  in ddgs_repr
        dms_matrix[idx, matrix_idx['idx_wt_rpr']:matrix_idx['idx_mut_rpr']] = \
                                                        mut_embed.cpu().numpy()
        
        # difference in representations
        diff_embed = mut_embed - self.wt_embed
        # add diff_repr to ddg
        dms_matrix[idx, matrix_idx['idx_mut_rpr']:matrix_idx['idx_diff_rpr']] \
                                                            =  diff_embed
        
        # add mut (integer) labels sequence 
        dms_matrix[idx, matrix_idx['idx_wt_seq']: matrix_idx['idx_mut_seq']\
                                                            = mut_seq  
        
        # save mut position
        dms_matrix[idx, matrix_idx['idx_pos'] = idx_mut
                   
        # save wt and mut aa label (integer vocab)
        dms_matrix[idx, matrix_idx['idx_wt'] = wt_aa
        dms_matrix[idx, matrix_idx['idx_mut'] = mut_aa
                   
        # add exp dms/ddg for the mutation 
        dms_matrix[idx, matrix_idx['idx_dms'] = dms_exp
                  
                 
        return dms_matrix
    
    def _get_mutation(self, row):
        '''takes index and row from dms/ddg file and returns
        XXX mutation sequence
        '''
        
        
        # get dms or ddgs experimental measurements
        dms_exp = float(row['DMS']) # obs first (0) is startcodon so counts from 1


        # get seq index of mutation
        idx_mut = row['RES'] # obs first (0) is startcodon so counts from 1

        # wt/ original aa in int
        wt_aa = utils.get_triAA_to_int(row['WT'])  

        #  which aa its mutated to , convert mutation from tri code to int 
        mut_aa = utils.get_triAA_to_int(row['MUT'])

        
        ## create mut seq from wt seq; first init np array from wt
        mut_seq = np.array(self.wt_seq, copy=True) 
        # change seq to include mutation 
        mut_seq[idx_mut] = mut_aa   # change aa in position
        
        # get mutational embedding
        mut_seq_OH = self.vocab_2_one_hot(mut_seq)
        mut_embed = self.model.embed(mut_seq_OH )

        return dms_exp, idx_mut, wt_aa, mut_aa, mut_seq, mut_embed
 
    
                   
    def split_train_test (self,dms_matrix):
        ''' randomly split in train and test set.
        Keep track of splitting by indexing
        '''

        shuffl_indx = np.arange(len(self.dms_file))
        np.random.shuffle(shuffl_indx)
        train_len = int(0.80 * len(self.dms_file))
        # random indexes
        train_indx = shuffl_indx [:train_len]
        test_indx = shuffl_indx [train_len:]

        # get train/test from random indx
        train = dms_matrix[train_indx]
        test = dms_matrix[test_indx]

        # jaja could have returned pandas df but so much trouble to do so  
        return  train, test


    
    @classmethod
    def vocab_2_one_hot(cls,integer_sequence):
        '''takes preprocessed fasta seq and converts to one hot for input to\
        CNN models
        '''
        seq = torch.tensor(integer_sequence,dtype=torch.long)
        
        if  torch.cuda.is_available(): 
            seq.cuda()
        # convert to one hot 
        seq_OH = utils.to_one_hot(seq)

        # convert for channels to come first
        seq_OH = torch.transpose(seq,0,1) #transpose dim 1,2 => channels=aa

        # add "batch" dimension=1 for model to work
        seq_OH = seq [None, :, :]
        
        return seq_OH 
            

                   
class HandleNNModel():
    '''deals with sequences and necessary conversions for using 
    the nn encoder/decoder
    '''
   
    @classmethod
    def convert_aa_to_int (cls, aa):
        aa_to_int = {
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
                   
        return aa_to_int[aa]
      
                  
                   
                   
    @classmethod
    def vocab_2_one_hot(cls,integer_sequence):
        '''takes preprocessed fasta seq and converts to one hot for input to\
        CNN models
        '''
        seq = torch.tensor(integer_sequence,dtype=torch.long)
        
        if  torch.cuda.is_available(): 
            seq.cuda()
        # convert to one hot 
        seq_OH = utils.to_one_hot(seq)

        # convert for channels to come first
        seq_OH = torch.transpose(seq,0,1) #transpose dim 1,2 => channels=aa

        # add "batch" dimension=1 for model to work
        seq_OH = seq [None, :, :]
        
        return seq_OH 

                   


class SaveDMS2json(PerformJSON):
    '''add mdoels downstream performance to  performance table:\
    downstream_performance.json.
    Args:
        nn_model: name of nn model architecture (str)
        model_params: dict of model_params
        
    '''
    def __init__(self, nn_model,  model_params, tape_task=None, dms_task=None, \
                  plots=None, paths = None, \
                 notes=None,debug=False):
        
        super().__init__()
    
        if debug:
            self.nn_model = self._rename_if_debug(nn_model)
        else:
            self.nn_model =self._add_model(nn_model, model_params)
                    
        self.model_params =  model_params
        self.rmsd = rmsd
        self.P_Corr = pearson
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
        if self.dms_task != None:
            Assertions.dms_task(self.dms_task)
            self.perform_dict[self.nn_model]['DMS'][self.dms_task] = {}
            self.perform_dict[self.nn_model]['DMS'][self.dms_task]['MSE'] = self.MSE
            self.perform_dict[self.nn_model]['DMS'][self.dms_task]['P_Corr'] = self.P_Corr

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
                nn_model += f"_ks{model_params['ks_conv']}"
                print(f'model architecture already in tabel, \
                        but not the same params, renaming model to \
                        {nn_model}',flush=True)
                            # check if model params in dict
                dict_model_params = self.perform_dict[nn_model]['MODEL_PARAMS']
            
                if model_params == dict_model_params:
                    print('Model already in table, might overwrite some data')
                    return True, nn_model

                else:
                    return False, nn_model 
        
        elif nn_model not in self.perform_dict.keys() :
            return False, nn_model

        
    

## Define amino acid to integer. Identical to Unirep conversion. 
## Obs! '23' all deleted in cleaning data
aa_to_int = {
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


             



# previously named get_pred()
def decoder_prediction(seq_input, model):
    '''get AE predictions on specific protein '''
    with torch.no_grad():        

        # convert to one hot 
        if len(seq_input.shape) <2:
            seq_input = torch.tensor(seq_input,dtype=torch.long)
            seq = utils.to_one_hot(seq_input)
            # convert for channels to come first
            seq = torch.transpose(seq,0,1) #transpose dim 1,2 => channels=aa
            # add "batch" dimension=1 for model to work
            seq = seq [None, :, :]
        
        else:
            seq = torch.tensor(seq_input,dtype=torch.float32)
        
        # to cuda 
        if  torch.cuda.is_available(): 
            model = model.cuda()
            seq = seq.cuda()
            
        # get wt representation for seq from encoder model
        out = model(seq)
        
        # get accuracy 
        _, predicted = torch.max(out.data, dim=1)
        
        
        # get correct with pad
        seq_input = seq_input[None, :]
        correct_w_pad = (seq_input == predicted)
        
        # filter out padding (0)
        msk = seq_input != 0
        pred_msk = predicted[msk]
        target_msk = seq_input[msk]
        correct = (target_msk == pred_msk)
        len_protein = msk.sum().item()
        
        return correct, correct_w_pad,len_protein


def visual_prot(pdb,seq_input, model,cam_orient=None):
    ''' visualises protein in cell output, where red in AE incorrect pred'''
    # get MDanalysis stuff
    u = mda.Universe(pdb, pdb)
    protein = u.select_atoms("segid A")
    start_resid = protein.resids[0]
    end_resid = protein.resids[-1]

    # get nn model predictions for sequence
    correct, correct_w_pad, len_protein = get_pred(seq_input, model)
    correct_pos = correct

    # CA = protein.select_atoms("name CA")
    # print('pdb',len(CA), 'start', start_resid, 'end',end_resid)

    # add column for colouring the predictions
    u.add_TopologyAttr('tempfactors')

    # loop through all atoms in given resid and give same tempfactor according to prediction
    temp_factors = []
    for index,res in enumerate(u.atoms.residues):
        resid = res.resid
        pred = correct_pos[resid]
        for atom in u.atoms.residues[index].atoms:
            temp_factors.append(pred)
    

    # add b-factors
    u.atoms.tempfactors = temp_factors
    protein = u.select_atoms("segid A") # only have chain A 

    # load universe object in nglview
    t = nv.MDAnalysisTrajectory(protein)
    w = nv.NGLWidget(t)

    # define represention
    w.representations = [
        {"type": "cartoon", "params": {
            "sele": "protein", "color": "bfactor"
        }},
        {"type": "ball+stick", "params": {
            "sele": "hetero"
        }}
    ]
    # w.download_image(filename='str2_ks5.png', factor=4, antialias=True, trim=False, transparent=True)
    if cam_orient != None:
        try: 
             w._set_camera_orientation(cam_orient)
        except:
            pass
  
    w._remote_call("setSize", target="Widget", args=["400px", "400px"])

    # use gui 
    # w.display(gui=True, use_box=False)

    # show in dislay
    return  w
    
    



def train_rand_forest(train_set, test_set, 
                      lat_space=500,
                      params=None,
                      n_trees = 120,  
                      train_w_labels=False, 
                      train_w_labels_only = False,
                      train_repr_diff=False):
    
    ls1 = lat_space # first :ls1 is wt representation
    ls2 = lat_space*2 # ls1:ls2 mut representation
    ls3 = lat_space*3 # ls2:ls3 = difference: wt repr - mut rep
    ls4 = lat_space*3 + 500 # ls2:ls3 is wt labels seq()
    ls5 = lat_space*3 + 2*500 # # ls2:ls3 is wt labels ()

    # define x and y for training. 
    y_train =train_set[:,-1] 
    y_test = test_set[:,-1]

    # which features to include in training together with repr
    if train_w_labels:
        x_train = np.concatenate((train_set[:,:ls2], train_set[:,ls3:ls5]),axis=1)
        x_test  = np.concatenate((test_set[:,:ls2], test_set[:,ls3:ls5]),axis=1) 
    elif train_w_labels_only: # only use labels in training, no repr
        x_train = train_set[:,ls4:ls5] 
        x_test  = test_set[:,ls4:ls5] 
    elif train_repr_diff: 
        x_train = train_set[:,ls2:ls3] 
        x_test  = test_set[:,ls2:ls3] 
    else: # only use representation
        x_train = train_set[:,:ls2] 
        x_test  = test_set[:,:ls2] 
    
    # init model using input params 
    if params is not None: 
        n_trees = params['n_estimators']
        min_samples_split = params['min_samples_split']
        min_samples_leaf = params['min_samples_leaf']
        max_features = params['max_features']
        max_depth = params['max_depth']
        bootstrap = params['bootstrap']
        rfr = RandomForestRegressor(n_estimators = n_trees, 
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    bootstrap=bootstrap, oob_score=bootstrap)

    else:     
        rfr = RandomForestRegressor(n_estimators = n_trees, oob_score=True)
    
    # train model     
    rfr.fit(x_train,y_train)
    # get model prediction  on test set
    y_pred = rfr.predict(x_test)
    # add pred to last column of original matrix
    test_set = np.column_stack((test_set, y_pred ))
 
    # loss/error function (MSE):
    rmsd = np.sqrt(np.mean((y_pred - y_test)**2 ))
    pearson = pearsonr(y_pred, y_test)
    
    # split in AE correctly predicted and incorrectly predicted
    AE_pred = test_set[:,-3] # col_AE_pred
    idx_cor = np.argwhere(AE_pred==1)
    idx_incor = np.argwhere(AE_pred==0)

    return rfr, test_set, idx_cor, idx_incor, rmsd, pearson

def CV_rand_forest(train_set, test_set, lat_space=500,
                    train_w_labels=False, train_w_labels_only = False,
                    train_repr_diff=False):
    
    ls1 = lat_space # first :ls1 is wt representation
    ls2 = lat_space*2 # ls1:ls2 mut representation
    ls3 = lat_space*3 # ls2:ls3 = difference: wt repr - mut rep
    ls4 = lat_space*3 + 500 # ls2:ls3 is wt labels seq()
    ls5 = lat_space*3 + 2*500 # # ls2:ls3 is wt labels ()

    # define x and y for training. 
    y_train =train_set[:,-1] 
    y_test = test_set[:,-1]

    # which features to include in training together with repr
    if train_w_labels:
        x_train = np.concatenate((train_set[:,:ls2], train_set[:,ls3:ls5]),axis=1)
        x_test  = np.concatenate((test_set[:,:ls2], test_set[:,ls3:ls5]),axis=1) 
    elif train_w_labels_only: # only use labels in training, no repr
        x_train = train_set[:,ls4:ls5] 
        x_test  = test_set[:,ls4:ls5] 
    elif train_repr_diff: 
        x_train = train_set[:,ls2:ls3] 
        x_test  = test_set[:,ls2:ls3] 
    else: # only use representation
        x_train = train_set[:,:ls2] 
        x_test  = test_set[:,:ls2] 
        

    # DEFINE FEATURES TO CV TEST 
    n_trees = np.arange(50,500,50)
    # Number of features to consider at every split
    max_feat = [ "auto",'sqrt','log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_trees,
                   'max_features': max_feat,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    ## Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    #  Fit the random search model
    rf_random.fit(x_train, y_train)
    print ('BEST PARAMS:')
    print(rf_random.best_params_)

 

    return rf_random.best_params_

                   