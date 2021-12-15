# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 12:42:34 2021

@author: owner
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import parsing_functions as pf
from scipy import signal
from tensorflow import keras
from keras import models
from keras import layers
import glob
from parse import *
import re
from numpy import expand_dims
import pandas as pd
import os.path
import pickle

#import package files
import run_data as RD
import bulbul_models_options as BM

"""
-----------------------------------------------------------------
    DATA PROCESSING
-----------------------------------------------------------------
"""

fs = 44100
# Set the frame parameters to be equivalent to the librosa defaults
# in the file's native sampling rate
dur = 1   # frames of 0.5 seconds
# frame_length = int(dur*fs) # (2048 * fs) // fs
# hop_length = int(frame_length/2) #(1024 * fs) // fs

nfft = 2048
hop = 700
n_filters = 50 # number of mel-filters
f_min = 1000 # mel filters low freq
f_max = 3500 # mel filters high freq
fL = 700 # low freq for bandpass filter
fH = 3900 # high freq for bandpass filter
Th = 0.35 # threshold for labling
min_xtrain = -80.0 # min value for dB spectrogram
data_path = 'data/train'
timeAug = False # if augmanttion is used

######-----option 1- run data_processing to generate training data-

######-----option 2- read pickle file from data directory if training exist-

if os.path.exists(data_path + "/large_df.pkl"):
            large_df = pd.read_pickle(data_path + "/large_df.pkl")
            df_aug = pd.read_pickle(data_path + "/df_aug.pkl")
else:
    if timeAug == True:
            large_df, df_aug = RD.data_processing(min_xtrain , data_path , 
                fs , dur ,  nfft , hop , n_filters  , f_min ,
                f_max , fL , fH , Th , timeAug)
            large_df.to_pickle(data_path + "/large_df" + '.pkl')
            df_aug.to_pickle(data_path + "/df_aug" + '.pkl')
    else:
            large_df = RD.data_processing(min_xtrain , data_path , 
                fs , dur ,  nfft , hop , n_filters  , f_min ,
                f_max , fL , fH , Th , timeAug)
            df_aug = np.array([])


runNetwork = True
generateLabels = False
runData = False
cutWords = False

#####-----run network-
# returned parameters-
# test_acc - accuracy, TPR - true positive rate, FPR - false positive rate
# FP_TP - false positive/true posotive
# TP, FP- number of true and false positive
if runNetwork:
    test_acc , TPR , FPR , FP_TP, TP, FP = RD.bulbul_conv_net(df = large_df ,
                dfaug = df_aug , train_prcnt = 0.7 , run_net = 'convnet')


"""
-----------------------------------------------------------------
    different options to use the model
-----------------------------------------------------------------
"""


#####-----generate labels to a txt file
##### if model is saved- no need to run network again. the function uses the model.
if generateLabels:
    label_df = BM.gen_labels(min_xtrain = -80, data_path = 'data/to_label', 
                fs = 44100, dur = 1,  nfft = 2048, hop = 700,
                n_filters = 50, f_min = 1000,
                f_max = 3500, fL = 700, fH = 3900 , Th_pred = 0.5)

    label_time = BM.gen_label_txt(label_df, frame_length = 1, hop_length = 0.5)

#####-----run model on new data
if runData:
    df_pred , dur_list , word_edge, files_tuples = BM.run_data(data_path = 'data/all' ,
                min_xtrain = -80.0 , fs = 44100, dur = 1 , nfft = 2048 ,
                hop = 700 , n_filters = 50 , f_min = 1000 , f_max = 3500)

####-----cut words and save in directory, count bulbul calls per hour- 
if cutWords:
    BM.cut_words(files_tuples , word_edge , path = 'data/all')



