# Representation learning on protein sequence CNNs

## Description: 
This project concerns representation learning using convolutional neural networks (CNN) on protein sequences (Uniref50).


It investigates performances of downstream tasks from both TAPE (i.e. stability or fluorescence) and and more random forrest trees on stability ddGs from protein G (Mayo) or DMS (Fowler) on pdb: 2H11 or pdb:1D5R

This project was small fun project to initially learn machine learning.

The results would likely hugely benefit from data curation of the uniref data as shown by:
- A. Rives et al.: https://www.biorxiv.org/content/10.1101/622803v3


## Results:
All results of this project are visualised in a streamlit app.
It visualises the results of all tested models, both decoder performance and downstream performances on TAPE tasks as well as individual proteins.

The streamlit app allows you too:
    - compare all models decoder performance
        - and distinguish between # layers, # fully connected etc
    - compare all models downstream performances on
        - TAPE Fluorescence task
        - TAPE Stability task
        - Single protein DMS mutations on pdb: 2H11 (Fowler)
        - Single protein DMS mutations on pdb: 1D5R (Fowler)
        - Single protein ddG mutations on protein G (Mayo)
    - investigate all individual trained models f.x. showing:
        - downstream prediction vs target on all above mentioned tasks
        - downstream prediction vs target histogram
        - downstream training plots
        - upstream training performance plot
        - upstream confusion matrix
        - upstream performance on N-,C- terminal ends
        - upstream prediction performance on each type of AA
        - show model architecture
        - show upstream/downstream log_files with all model parameters


To run streamlit app:
Just ensure you have streamlit installed in your conda environment see: `environment_streamlit.yml`
and then run:

```
streamlit run streamlit_show_all_project_results.py
```
This will open a streamlit port. Use the same approach as with jupyter notebooks to open it. I.e. use to port number given and opeen it in your browser: 
```
http://localhost:8501
```
If you run the streamlit app on a remote server, then use the same approach as for notebooks running remotely, e.g. open a tunnel to the remote      server from your local sever by:
```
ssh -N -L 8501:localhost:8501 <remote_server_name>
```
and then open the tunneled port in your browser again using `http://localhost:8501`

