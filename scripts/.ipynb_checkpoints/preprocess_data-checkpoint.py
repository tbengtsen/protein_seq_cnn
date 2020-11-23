"""
Preprocessing of Uniref50 from Unirep. 
"""

import numpy as np
import pickle as pkl
import sys
import h5py

def fasta_to_input_format(in_file,
                         o_name = 'preproccessed', o_type = 'h5', max_l = 498, test = 0.1, valid = 0.02):
    '''
    Preproccess uniref50 fasta file.
    	preprocess by:
            - remove all fasta headers, each line is a full protein seq
            - remove all seq with Ambiguous aa [BJXZ]
            - remove all seq longer than max_l
	    - convert all aa to integers (see table below) 
            - splits into train, test, validation datasets seperatedly
	
	Saves npy where rows = sequences 
    '''
    formatted_train = []
    formatted_test = []
    formatted_valid = []

    with open(in_file, 'r') as f:

        seq = ""
        for line in f:
            # add to protein seq
            if not line[0] == '>':
                seq += line.replace("\n","")

            ## save full protein sequence to file ##
            if line[0] == '>' and not seq == "":
                print (line)

                ## check if max len of protein seq exceeded or ambiguous aa ##
                if max_len(seq, max_l) or ambiguous_aa(seq):
                    seq ="" # discard seq

                else:
                    ## convert to int ##
                    int_seq = aa_seq_to_int(seq)

                    ## pad seq with 0's ##
                    int_seq = pad_seq(int_seq, max_l)
                    
                    ## seperate into train, valid and test sets ##
                    rand_nr = np.random.uniform()
                    if rand_nr < test:
                        formatted_test.append(int_seq)

                    elif rand_nr >test and rand_nr < (test+valid):
                        formatted_valid.append(int_seq)

                    else:
                        formatted_train.append(int_seq)

                    
                    ## empty str for next protein in fasta file ##
                    seq = ""

    
    ## save formatted seqs ##
    if o_type is 'h5' :
        save_chunks_hdf5 (formatted_train, formatted_test, formatted_valid, o_name)

    elif o_type is 'npz':
        # numpy save compressed matrix
        save_npz (formatted_train, formatted_test, formatted_valid, o_name)
    else: 
        save_chunks_hdf5 (formatted_train, formatted_test, formatted_valid, o_name)
        raise ValueError("Output file type argument does not exist. o_type,\
                must be either 'h5' (HDF5) or 'npz' (numpy compressed dict). \
                Proceeds by saving to h5")

def max_len (seq, max_l=500 ):
    '''
    check if exceeds max lenght of protein seq 
    '''
    if len(seq) > max_l:
        return True
    else:
        return False


def ambiguous_aa(seq):
    '''
    Check if ambiguous amino acid. 
    '''
    if 'X' in seq: # Unknown aa
        return True
    elif 'Z' in seq: # ambiguous Glutamic acid or GLutamine
        return True
    elif 'B' in seq: # ambiguous Asparagine or aspartic acid
        return True
    elif 'J' in seq: # ambiguous Leucine or isoleucine
        return True 
    else: 
        return False 



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


def aa_seq_to_int(seq):
    """
    Return the sequence converted to integers as a list 
    plus start(23) and stop(24) integers. From Unirep. 
    """
    return [23] + [aa_to_int[aa] for aa in seq] + [24]

def pad_seq(seq, max_l):
    '''
    Pads the integer sequence with 0's up to max_l+2 
    '''
    padded_seq = [0]*(max_l+2) # +2 as start and stop added to seq 
    padded_seq[:len(seq)] = seq
    
    return padded_seq


def save_npz (formatted_train, formatted_test, formatted_valid, out):
    ''' save to individual npz matricesm 
    splits train to multiple random npz matrices
    '''
    # test set 
    formatted_test = np.array(formatted_test)
    np.savez_compressed(out+'_test.npz', test=formatted_test)

    # valid set 
    formatted_valid = np.array(formatted_valid)
    np.savez_compressed(out+'_valid.npz', valid=formatted_valid)
    
    # train set
    formatted_train = np.array(formatted_train)
    np.random.shuffle(formatted_train) # randomize to save chunks, modifies in-place
    formatted_train = np.array_split(formatted_train, 10)
    for i, data in formatted_train:
        np.savez_compressed(out + "_train_random{}.npz".format(i), train=formatted_train)




def save_chunks_hdf5 (formatted_train, formatted_test, formatted_valid, out):
    '''
    Saves matrices in hdf5 file. 
    Splits train and test set into chunks to save in smaller matrices 
    for loading into memory. Random shuffles matrices before splitting 
    to chunks. 
    '''
    # open file for saving
    f = h5py.File(out+'.h5', 'w') 
    
    # convert to numpy
    formatted_train = np.array(formatted_train)
    formatted_test = np.array(formatted_test)
    formatted_valid = np.array(formatted_valid)

    # randomize to save chunks obs modifies in place
    np.random.shuffle(formatted_train) # randomize to save chunks, modifies in place
    np.random.shuffle(formatted_test) # randomize to save chunks 

    # split into chunks
    formatted_train = np.array_split(formatted_train, 10) # split in chunks to save
    formatted_test = np.array_split(formatted_test, 2) # split in chunks to save

    # create groups/subdirectories in hdf5 file
    trainset = f.create_group("train")
    testset = f.create_group("test")
    validset = f.create_group("valid")
    
    # save in subdir
    for i, data in enumerate(formatted_train):
        ds = trainset.create_dataset("train_{}".format(i), data=data , dtype='i8')
    
    for i, data in enumerate(formatted_test):
        ds = testset.create_dataset("test_{}".format(i), data=data , dtype='i8')

    ds = validset.create_dataset("valid", data=formatted_valid , dtype='i8')

    # close 
    f.close()



if __name__=="__main__":

    # I/O files
    # test files uniref50_1000lines.fasta
    #uniref50  = "../data/uniref50_500000lines.fasta" # small scale testing file
    #o_name = '../data/uniref_prep/uniref_500000lines_preprocessed'

    # all uniref files
    uniref50  = "../data/uniref50.fasta" # downloaded uniref
    o_name = '../data/uniref_prep/uniref50_preprocessed_ny'

    # clean fasta then save train/test/valid as npz's or h5
    fasta_to_input_format(uniref50, o_name, o_type='h5', max_l=498, test=0.1,valid=0.02) 
