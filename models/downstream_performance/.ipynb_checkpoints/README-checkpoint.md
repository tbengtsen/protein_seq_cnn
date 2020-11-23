# PERFORMANCE OF PRETRAINED MODELS (REPRESENTATIONS) ON TAPE

All pretrained models are defined in models/ directory.
**see script `tape_evaluate.py` for how these were run. **

preformance.json has the following architecture:
```
dict_keys(['MODEL_NAME'])
    key:  TAPE
	   key:  stability {'MSE': None, 'MAE': None, 'S_Corr': None}
	   key:  fluorescence {'MSE': None, 'MAE': None, 'S_Corr': None}
    key:  DMS
	   key:  protein_g {'MSE': None, 'MAE': None, 'S_Corr': None}
	   key:  2H11 {'MSE': None, 'MAE': None, 'S_Corr': None}
	   key:  1D5R {'MSE': None, 'MAE': None, 'S_Corr': None}
    key:  MODEL_PARAMS
	   key:  latent_size None
	   key:  layers None
	   key:  fully_con None
	   key:  compression_seq None
	   key:  channels None
	   key:  ks_conv None
	   key:  str_conv None
	   key:  pad_conv None
	   key:  ks_pool None
	   key:  str_pool None
	   key:  pad_pool None
	   key:  LOGFILE None
    key:  PATHS
	   key:  model_dir None
	   key:  tape_dir None
	   key:  DMS_dir None
    key:  PLOTS
	   key:  stability None
	   key:  fluorescence None
	   key:  protein_g None
	   key:  2H11 None
	   key:  1D5R None
 ```