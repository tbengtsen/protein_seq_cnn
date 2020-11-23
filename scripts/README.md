# SCRIPTS FOR SEQUENCE AUTOENCODER
## Contains:
### Data Processing scripts:
    - preprocess_data.py: 
        Script for preprocessing uniref50 data
        See dir: representation_learning/data/uniref for more info
    
    - preprocess_w_taxon.py:
        Script for preprocessing uniref50 data using only sequences in specified taxon family
        See dir: representation_learning/data/uniref/uniref_by_taxonomi for more info 
        OBS! ONLY WORKS IN PYTHON2.7 due to module library for getting evolution
    

### CNN train scripts 
    - cnn_seq_train.py: 
       Script to launch training of CNN models, takes many inputs among others which nn_model to use, see --help for info. 
    
    - submit_training.sh:
        Slurm submit script to launch cnn_seq_train.py for CNN training with parser arguments.
### Dwonstream training on TAPE tasks
    - tape_evaluate.py: 
       Script to launch downstream training and evaluation of tape tasks using representations from upstream pre-trained CNN models. 
       Takes many inputs among others which nn_model to use, see --help for info 
    
    - submit_tape.sh
        Slurm submit script to launch tape_evaluate.py for downstream evaluation of pre-trained CNN representation models. Takes parser arguments.
    
### Directories  
    - utils/
        directory with all utils for CNN training and downstream training. Contains:
            - utils.py:
                script with all utilitily functions/classes for use in cnn_seq_train.py.
            - metrics.py
                training performance metrics for outputs as well as test/validation performance and outputs text/plots. 
            - tape_cnn.py
                utils for downstream training and evaluation of tape tasks on representation from CNN model
             - ddgs.py
                 utils for downstream training and evaluation  of ddgs predictions on representation from CNN model


    - nn_models/:
        Directory which contains all made architectures of CNN on sequences. 
    
    - pytorchtape/:
        Nicki's pytorch interface to TAPE's tensorflow.
    
    - streamlit_apps/:
        containing the streamlit app python file to show all results in a streamlit app, see main README.md file for how to run it. 
    
