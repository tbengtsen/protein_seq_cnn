# NN Convolutional models used for training. 
### Using either a stride or average pool for compressing sequence. 

To be imported for training using the option --model in submission file. Obs do not use ending ".py" in input. 
E.g. python cnn_seq_train.py --f ../data/uniref/uniref50_preprocessed.h5 --model ConvStride_w_FC ......

## Models:  

### average pooling
-----------------

cnn_avgpool_ls500_0fc_seq4
cnn_avgpool_ls500_1fc_seq63
cnn_avgpool_ls400_0fc_seq8
cnn_avgpool_ls400_0fc_seq4

### strided convolution
-----------------
cnn_strided_ls500_1fc_seq63
cnn_strided_ls500_2fc_seq63
cnn_strided_ls400_0fc_seq8
cnn_strided_ls500_0fc_seq4
cnn_strided_ls512_0fc_seq32
cnn_strided_ls504_0fc_seq63
cnn_strided_ls504_0fc_seq63_ksvariable

### compressing of channels only
------------------------------
cnn_autoencode_AA



