"""
Preprocessing of uniref data using taxons to remove some genes
Only works with python2.7 
"""
import numpy as np
import pickle as pkl
# taxon module
import sys
import h5py
from ete3 import NCBITaxa
ncbi = NCBITaxa()


def fasta_to_input_format(
    in_file,
    o_name,
    o_type="h5",
    max_l=500,
    test=0.1,
    valid=0.1,
    incl_all_taxa="Mammalia",
):
    """
    Preproccess uniref50 fasta file.
        preprocess by:
            - remove all fasta headers, each line is a full protein seq
            - remove all seq with Ambiguous aa [BJXZ]
            - remove all seq longer than max_l
            - remove all seq that are not in input taxon lineage
            - convert all aa to integers (see table below)
            - pads seq up ot max_l by 0's

        Saves npy where rows = sequences
    """

    formatted_train = []
    formatted_test = []
    formatted_valid = []
    nr = 0
    with open(in_file, "r") as f:
        seq = ""
        for line in f.readlines():
            # add to protein seq
            if not line[0] == ">":
                seq += line.replace("\n", "")

            # save full protein sequence to file
            if line[0] == ">" and not seq == "":
                nr += 1

                ## CLEAN DATA ##
                # discard if max len of protein seq exceeeded
                if max_len(seq, max_l):
                    seq = ""  # discard seq

                # discard if ambiguous/undetermined aa (B,J,Z,X)
                elif ambiguous_aa(seq):
                    seq = ""  # discard seq

                # discard if cluster includes organisms higher in taxonomy
                elif discard_by_taxonomy(line, incl_all=incl_all_taxa):
                    seq = ""  # discard seq

                ## CONVERT DATA ##
                else:
                    # convert to int
                    int_seq = aa_seq_to_int(seq)

                    # pad seq with 0's
                    int_seq = pad_seq(int_seq, max_l)

                    ## seperate into train, valid and test sets ##
                    rand_nr = np.random.uniform()
                    if rand_nr < test:
                        formatted_test.append(int_seq)

                    elif rand_nr > test and rand_nr < (test + valid):
                        formatted_valid.append(int_seq)

                    else:
                        formatted_train.append(int_seq)

                    # empty str for next protein in fasta file
                    seq = ""

                # save chkpoints as keep crashing
                if nr % 500000 == 0:
                    ## save formatted seqs ##
                    if o_type is "h5":
                        chkp_name = o_name + "_chkp_{}seqs".format(nr)
                        save_chunks_hdf5(
                            formatted_train, formatted_test, formatted_valid, chkp_name
                        )

                    elif o_type is "npz":
                        # numpy save compressed matrix
                        save_npz(
                            formatted_train, formatted_test, formatted_valid, o_name
                        )

                    ## save formatted seqs ##
                    if o_type is "h5":
                        save_chunks_hdf5(
                            formatted_train, formatted_test, formatted_valid, o_name
                        )

                    elif o_type is "npz":
                        # numpy save compressed matrix
                        save_npz(
                            formatted_train, formatted_test, formatted_valid, o_name
                        )

                    else:
                        save_chunks_hdf5(
                            formatted_train, formatted_test, formatted_valid, o_name
                        )
                        raise ValueError(
                            "Output file type argument does not exist. o_type,\
                                must be either 'h5' (HDF5) or 'npz' (numpy compressed dict). \
                                Proceeds by saving to h5"
                        )


def max_len(seq, max_l=500):
    """
    check if exceeds max lenght of protein seq
    """
    if len(seq) > max_l:
        return True
    else:
        return False


def ambiguous_aa(seq):
    """
    Check if ambiguous amino acid.
    """
    if "X" in seq:  # Unknown aa
        return True
    elif "Z" in seq:  # ambiguous Glutamic acid or GLutamine
        return True
    elif "B" in seq:  # ambiguous Asparagine or aspartic acid
        return True
    elif "J" in seq:  # ambiguous Leucine or isoleucine
        return True
    else:
        return False


def discard_by_taxonomy(fasta_header, incl_all):
    """
    Whether to include or discard cluster by taxonomy rank.
    F.x. include all mammals only.
    takes: fasta header string name of gene and top rank to include (e.g. mammals)
    returns False/True if in/not in line of ascendents (True if to be discarded)
    """

    # get taxid of top rank/level to include
    name_top_rank = ncbi.get_name_translator([incl_all])
    taxID_top_rank = name_top_rank[incl_all][0]

    # extract taxonomy ID of current cluster from header
    # taxon ID is the lowest common taxon shared by all members in the cluster
    taxID = [i for i in fasta_header.split() if "TaxID=" in i]
    try:
        taxID = int(taxID[0].split("TaxID=")[-1])
        print fasta_header.strip()

        # get all ascendents/inheritance of that gene
        try:
            lineage = ncbi.get_lineage(taxID)
            # check if top rank in line of ascendents otherwise discard
            if taxID_top_rank in lineage:
                return False  # do not discard seq
            else:
                return True  # do discard seq

        except:
            return True  # discard seq

    except:
        return True  # discard seq


## Define amino acid to integer. Identical to Unirep conversion.
## Obs! '23' all deleted in cleaning data
aa_to_int = {
    "M": 1,
    "R": 2,
    "H": 3,
    "K": 4,
    "D": 5,
    "E": 6,
    "S": 7,
    "T": 8,
    "N": 9,
    "Q": 10,
    "C": 11,
    "U": 12,  # Selenocystein.
    "G": 13,
    "P": 14,
    "A": 15,
    "V": 16,
    "I": 17,
    "F": 18,
    "Y": 19,
    "W": 20,
    "L": 21,
    "O": 22,  # Pyrrolysine
    "start": 23,
    "stop": 24,
}


def aa_seq_to_int(seq):
    """
    Return the sequence converted to integers as a list
    plus start(23) and stop(24) integers. From Unirep.
    """
    return [23] + [aa_to_int[aa] for aa in seq] + [24]


def pad_seq(seq, max_l):
    """
    Pads the integer sequence with 0's up to max_l+2
    """
    padded_seq = [0] * (max_l + 2)  # needs space for start/ stop
    padded_seq[: len(seq)] = seq

    return padded_seq


def save_npz(formatted_train, formatted_test, formatted_valid, out):
    """save to individual npz matricesm
    splits train to multiple random npz matrices
    """
    # test set
    formatted_test = np.array(formatted_test)
    np.savez_compressed(out + "_test.npz", test=formatted_test)

    # valid set
    formatted_valid = np.array(formatted_valid)
    np.savez_compressed(out + "_valid.npz", valid=formatted_valid)

    # train set
    formatted_train = np.array(formatted_train)
    np.random.shuffle(formatted_train)  # randomize to save chunks, modifies in-place
    formatted_train = np.array_split(formatted_train, 5)
    for i, data in formatted_train:
        np.savez_compressed(
            out + "_train_random{}.npz".format(i), train=formatted_train
        )


def save_chunks_hdf5(formatted_train, formatted_test, formatted_valid, out):
    """
    Saves matrices in hdf5 file.
    Splits train and test set into chunks to save in smaller matrices
    for loading into memory. Random shuffles matrices before splitting
    to chunks.
    """
    # open file for saving
    f = h5py.File(out + ".h5", "w")

    # convert to numpy
    formatted_train = np.array(formatted_train)
    formatted_test = np.array(formatted_test)
    formatted_valid = np.array(formatted_valid)

    # randomize to save chunks obs modifies in place
    np.random.shuffle(formatted_train)  # randomize to save chunks, modifies in place
    np.random.shuffle(formatted_test)  # randomize to save chunks

    # split into chunks
    formatted_train = np.array_split(formatted_train, 5)  # split in chunks to save
    formatted_test = np.array_split(formatted_test, 1)  # split in chunks to save

    # create groups/subdirectories in hdf5 file
    trainset = f.create_group("train")
    testset = f.create_group("test")
    validset = f.create_group("valid")

    # save in sub-directories in hdf5
    for i, data in enumerate(formatted_train):
        ds = trainset.create_dataset("train_{}".format(i), data=data, dtype="i8")

    for i, data in enumerate(formatted_test):
        ds = testset.create_dataset("test_{}".format(i), data=data, dtype="i8")

    ds = validset.create_dataset("valid", data=formatted_valid, dtype="i8")

    # close
    f.close()


if __name__=="__main__":

    # I/O files
    uniref50  = "../uniref50.fasta" # downloaded uniref
    o_name = 'uniref50_preprocessed_Bacteria'

    # choose f.x.: Eukaryota or Mammalia
    # clean fasta and pad seq then save train/valid/test(10%) as npz's
    fasta_to_input_format( 
        uniref50, 
        o_name,
        o_type = 'h5',
        max_l = 498,
        test = 0.1,
        valid = 0.05,
        incl_all_taxa = 'Bacteria')