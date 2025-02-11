#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Don"t show debug info while training
from seq_ml.model import main_function
    
config = {
    # DEFAULT TRAINING CONFIG
    "out_dir": "/ceph/hpc/home/novljanj/data_storage/projects/smOOPS_paper/Analysis/Exploratory/Model_Training/Models/Testing_models", # The output directory for the whole training run, if it doesn't exist it will be created
    "full_file": "/ceph/hpc/home/novljanj/data_storage/projects/smOOPS_paper/Data/machine_learning_input_prep/naive_transcripts_with_fasta_clip_m6a_paris_intra_paris_inter_postar3_rnafold_length_control_collapsed.bed", # The path to the full dataset file
    "sequence_column_name": 'sequence', # The name of the column containing the sequences (else None)
    "signal_column_name": ['global_iclip'  ,  'm6a'  ,   'paris_intramol' , 'paris_intergroup'    ,    'paris_intragroup'   ,     'AGO2_postar3' ,   'APC_postar3' ,    'CBP_postar3',     
                           'CELF_postar3'  ,  'CELF1_postar3' ,   'CELF2_postar3' ,  'CELF4_postar3' ,  'CIRBP_postar3' ,  'CPSF6_postar3' ,  'CREBBP_postar3' , 'ELAVL1_postar3',  'EZH2_postar3',
                           'FAM120A_postar3', 'FMR1_postar3' ,   'FUS_postar3'  ,   'HNRNPR_postar3',  'LIN28A_postar3' , 'MBNL1_postar3'  ,  'MBNL2_postar3' ,  'MBNL3_postar3' ,  'MSI2_postar3'   , 
                           'NOVA1_postar3' ,  'NOVA2_postar3'  , 'PABPC1_postar3' , 'POU5F1_postar3' , 'PTBP1_postar3'  , 'PTBP2_postar3'  , 'RBFOX1_postar3'  ,'RBFOX2_postar3' , 'RBFOX3_postar3'  ,
                           'RBM10_postar3' ,  'RBM3_postar3'   ,  'SRRM4_postar3' ,  'SRSF1_postar3' ,  'SRSF2_postar3' ,  'SRSF3_postar3'  , 'SRSF4_postar3' ,  'SRSF7_postar3' ,  'TAF15_postar3'   ,
                           'TARDBP_postar3' , 'TTP_postar3'   ,  'U2AF2_postar3'  , 'UPF1_postar3'  ,  'YTHDC2_postar3' , 'YY1_postar3'    ,  'ZFP36_postar3'  , 'rnafold'], # List of columns containing the signal (else None), the signal should be a comma sepatared string of numbers: 0.5,1.2,0.8... ##Dev note: The code for the sparse encoding is still present, and it could be implemented as an alternative, however I think the full encoding is more intuitive for general use.## 
    "group_column_name": "smoops_naive", # The name of the column containing the group labels
    "sep": "\t", # The separator used in the dataset file
    "test_and_validation_size": 0.3, # The proportion of the dataset to use for testing and validations
    "fix_train_val_split": True, # If True, the train and validation sets will be fixed, if False, they will be randomised each run (test dataset will always be fixed). Optimising run overrides this and fixes the dataset.
    
    "batch_size": 32, # The batch size for training
    
    "num_blocks": 3, # The number of convolution blocks in the model
    "dropout_prob": 0.1, # The dropout probability
    "l1_lambda": 0.0001, # The l1 regularization lambda
    "reduce_by": 3, # The factor by which to reduce the sequence length in each block
    "Final_CNN_units": 512, # The number of units in the first convolutional layer
    "CNN_Units_Increase_By_Percent": 0.25, # The percentage by which to reduce the number of units in each convolution block
    "Increase_Kernel_By": 1, # The factor by which to increase the kernel size in each subsequent block, 1 is no increase
    "Increase_Dilation_By": 1, # The factor by which to increase the dilation in each subsequent block, 1 is no dilation
    "LSTM_units": 512, # The number of units in the LSTM layer
    "Dense_units": 512, # The number of units in the dense layer
    "kernel_size": 5, # The kernel size for the convolutional layers
    "after_cnn_normalization": None, # Takes the type of normalisation after each convolution block:  None, "BatchNormalization", "LayerNormalization"
    "after_lstm_normalization": None, # Takes the type of normalisation after each convolution block:  None, "BatchNormalization", "LayerNormalization"
    "learning_rate": 1e-04, # The learning rate for the Adam optimizer
    "rnn_type": "LSTM", # The type of RNN to use: "LSTM", "GRU"
    
    "epochs": 10000, # The number of epochs to train for
    "patience": 20, # The number of epochs to wait if there is no improvement to the validation metric
    "repeat_training_for": 1, # The number of times to repeat the training for.
    
    # OPTIMISING
    "OPTIMISE": False, # Set to true to optimise the hyperparameters
    "n_trials": 60, # Number of trials in the optimisation - more trials are needed for more hyperparameters (use as many as you can afford in time and compute)
    "hyperparameters_to_optimise": ["num_blocks", "dropout_prob", "l1_lambda", "reduce_by", "Final_CNN_units", "CNN_Units_Increase_By_Percent", "Increase_Kernel_By", "Increase_Dilation_By", "LSTM_units", "Dense_units", "kernel_size", "learning_rate", "after_cnn_normalization", "after_lstm_normalization", "rnn_type"] # The hyperparameters to optimise
}

if __name__ == "__main__":
    main_function(config)