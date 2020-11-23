

# downstream evaluate on flourence task 
python tape_evaluate.py --architecture cnn_strided_ls500_1fc_seq63.py --tape_task fluorescence --trained_model ../models/cnn_strided_ls500_1fc_seq63/str2_ks11/checkpoint_model_epoch1_train_ds15.pt --kernel_size 11 --stride 2 --padding 5 --batch_size 1000 --lr 0.0001 --epochs 300

# downstream evaluate on stability task
python tape_evaluate.py --architecture cnn_strided_ls500_1fc_seq63.py --tape_task stability --trained_model ../models/cnn_strided_ls500_1fc_seq63/str2_ks11/Restart/checkpoint_model_epoch4_train_ds19.pt --kernel_size 11 --stride 2 --padding 5 --batch_size 1000 --lr 0.0001 --epochs 400
