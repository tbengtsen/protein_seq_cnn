# -*- coding: utf-8 -*-
"""
@author: Tone Bengtsen
"""
from .utils import Prepare_Data
from .utils import LogFile 
from .utils import save_checkpoint 
from .utils import load_checkpoint 
from .utils import save_final_model
from .utils import load_final_model
from .utils import to_one_hot
from .utils import from_one_hot
from .utils import get_aa_to_int
from .utils import get_int_to_aa
from .utils import int_to_aa
from .utils import get_triAA_to_int
# metrics of cnn learning
from .metrics import PerformMetrics
from .metrics import TrainingMetrics
# utils for cnn interface with nicki's pytorchtape
from .tape_cnn import TapeMetrics
from .tape_cnn import TapeOutputs
from .tape_cnn import SavePerform2json
from .tape_cnn import RepresentationModel
from .tape_cnn import Tape2CNN


# from .ddgs import ddg_fasta_2_input
# from .ddgs import get_repr_from_encoder
# from .ddgs import decoder_prediction
# from .ddgs import repr_ddg
# from .ddgs import train_rand_forest
# from .ddgs import CV_rand_forest
# from .ddgs import get_ddgs_files