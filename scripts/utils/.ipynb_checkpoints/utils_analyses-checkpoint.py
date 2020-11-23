import os
import numpy as np
import torch
from torchsummary import summary



def seq_pos_acc(correct, msk, N_term, C_term):
    ''' get position specific accuracy. For N-term  and C-term.  '''

    # From N term
    correct_msk = np.logical_and(correct.cpu(), msk.cpu()) #msk out padding
    N_term += torch.sum(correct_msk, dim=0) # sum ovre columns 
    
    # From C-term (bckwards)
    bckwrds_indx = torch.sum(msk_padding, dim=1) #find lenght index of each protein
    bckwrds_indx += -21 # 
    try: # in case protein smaller than 21 aa
        bckwrds = pos_bck(correct_msk, bckwrds_indx, num_elem=20)
        C_term += torch.sum(bckwrds , dim=0)
    except: 
        pass

    
    return N_term, C_term


def pos_bck(matrix, indx, num_elem=20):
    '''takes a matrix and return new matrix of shape: m,n = A.shape[0], num_elem
    Where each row corresponds to  A but with len=num_elem and starting from the 
    index defined in indx. Used in  '''
    all_indx = indx[:,None] + torch.arange(num_elem)
    return matrix[np.arange(all_indx.shape[0])[:,None], all_indx]
