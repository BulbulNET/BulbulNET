# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:30:34 2021

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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras as keras
import pandas as pd
import glob
from parse import *
import re
from numpy import expand_dims
import os.path
import pickle

import Utils as Ut
import bulbul_models_options as BM


def data_processing(min_xtrain = -80.0 , data_path = 'data/train', 
               fs = 44100, dur = 0.5,  nfft = 2048, hop = 700,
               n_filters = 50, f_min = 1000,
               f_max = 3500, fL = 700, fH = 3900 , Th = 0.35 , timeAug = True):
    labelsPath = data_path
    # Set the frame parameters to be equivalent to the librosa defaults
    # in the file's native sampling rate
    frame_length = int(dur*fs) # (2048 * fs) // fs
    hop_length = int(frame_length/2) #(1024 * fs) // fs
    
    # get  list of all files  within labelsPath
    wav_files=glob.glob(labelsPath + "/*.wav", recursive=True)
    txt_files=glob.glob(labelsPath + "/*.txt", recursive=True)
    N = len(wav_files)
    
    ##lists for dataframe:
    MFBmat = []
    bin_labels = []
    spectrograms = [] #option for saving spectrogram
    large_df = pd.DataFrame([])
    df_aug = pd.DataFrame([])
    
    for q in range(N):
        labels_tuples = pf.read_lable_file(str(txt_files[q]), "")
        labels = [(int(fs*float(i[0])), int(fs*float(i[1]))) for i in labels_tuples]        
       # print(labels)    
    
        #end = [int(float(i[1])) for i in labels_tuples]
        L = len(labels)
        # Stream the data, working on 128 frames at a time
        fs = librosa.get_samplerate(str(wav_files[q]))
        stream = librosa.stream(str(wav_files[q]),
                                block_length=1,
                                frame_length=frame_length,
                                hop_length=hop_length)        
        data_files = []
        data_files_aug = []
        ntime_start = 0
        j = 0
        for y in stream:
            if y.size == frame_length:

               ## function 1 - read wav and label- find 0 and 1 lables
               bin_l = BM.read_wav_label(frame_length , j , hop_length ,
                                         labels , L , Th , bin_l = 0)
     
               ## function 2 - play file
               # BM.play_file(y , fs)
                   
               ##calculate mel spectogram
               MFB = librosa.feature.melspectrogram(y , sr = fs, 
                         n_fft = nfft,
                         hop_length = hop, n_mels=n_filters,       
                         fmax=f_max, fmin = f_min)
               MFB_med = Ut.medclip(MFB, 3.5, 1e-14)
               MFB_dB = librosa.power_to_db(MFB_med, ref=np.max)
               MFBclean_dB = Ut.blobRemove(MFB_dB, 3, floordB = -80, Thresh = -60 )

               # time stretch and putch shift augmentation
               if timeAug:
                      stretch_factor = np.random.uniform(0.7, 1.3)                    
                      y_aug = librosa.effects.time_stretch(y, stretch_factor)
                      if y_aug.size > y.size :
                          y_aug = y_aug[:y.size]
                      elif y_aug.size < y.size:
                          Lgap = y.size - y_aug.size
                          gap = y[y.size-Lgap:y.size] # padding with the right hand side of y
                          y_aug = np.concatenate((y_aug , gap) , axis = 0)
                                            
                      pm = (np.random.randint(0,2) -0.5)*2
                      n_steps = pm*np.random.randint(0,3,())
                      librosa.effects.pitch_shift(y_aug, fs, n_steps, bins_per_octave=24)   

                      MFB_aug = librosa.feature.melspectrogram(y_aug , sr = fs, 
                         n_fft = nfft,
                         hop_length = hop, n_mels=n_filters,       
                         fmax=f_max, fmin = f_min)
                      MFB_med_aug = Ut.medclip(MFB_aug, 3.5, 1e-14)
                      MFB_dB_aug = librosa.power_to_db(MFB_med_aug, ref=np.max)
                      MFBclean_dB_aug = Ut.blobRemove(MFB_dB_aug, 3, floordB = -80, Thresh = -60 )
            
                      data_files_aug.append([q , j , bin_l , MFBclean_dB_aug])
                      column = ['index_dir' , 'index_file' , 'label' , 'MFB']
                      

               
               data_files.append([q , j , bin_l , MFBclean_dB])
               column = ['index_dir' , 'index_file' , 'label' , 'MFB']
               j = j + 1
           
            
        df = pd.DataFrame(data_files , columns = column)
    
        large_df = pd.concat([large_df,df], ignore_index=True)
        
        if timeAug:
            dfA = pd.DataFrame(data_files_aug , columns = column)
    
            df_aug = pd.concat([df_aug,dfA], ignore_index=True)
            
        
    return large_df , df_aug #, MFBmat, bin_labels, predictions

def add_noise(x):
    '''Add random noise to an image'''
    var = 5.0
    deviation = var*np.random.random()
    noise = np.random.normal(0, deviation, x.shape)
    y = x + noise
    y = np.clip(y, -80., 0.)
    return y

 
def data_augmentation(x_train, augment_factor = 0.35):
    data_aug = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
        
    layers.experimental.preprocessing.RandomZoom(height_factor=(-0.2, -0.01), 
            width_factor = (-0.2, -0.01), fill_mode='reflect', interpolation='bilinear' )])
    
    X_train_Aug = [] 
    for ind in range(x_train.shape[0]):
    #         randnum = np.random.rand()
    #         augmented_images = data_aug(x_train[ind])
    #         X_train_Aug.append(augmented_images)
        randnum = np.random.rand()
        if randnum > augment_factor:
            augmented_images = data_aug(x_train[ind])
            X_train_Aug.append(augmented_images)  
        else:
            augmented_images = add_noise(x_train[ind])
            X_train_Aug.append(augmented_images)
            
                       
########----check Augmentation-----######
        # plt.subplot(211)
        # plt.imshow(augmented_images[0])
        # plt.subplot(212)
        # orig_spect = x_train[ind]
        # plt.imshow(orig_spect[0,:,:,:])
        # plt.pause(1) 
        
    return X_train_Aug

def bulbul_conv_net(df , dfaug , train_prcnt = 0.7, run_net = 'convnet', augmentation = True):
    bin_labels = df['label'].tolist()
    MFBmat = df['MFB'].tolist()
    train_size = len(MFBmat)
    train_index = int(train_prcnt*train_size)      

    if dfaug.size != 0:
        MFB_tpaug = dfaug['MFB'].tolist()
        x_train_tp_aug = MFB_tpaug[:train_index]
        x_train_tp_aug = expand_dims(x_train_tp_aug, 3)
        x_train_tp_aug = expand_dims(x_train_tp_aug, 1)
  
   
    x_train_orig = np.asarray(MFBmat[:train_index]).astype('float32')
    x_test_orig = np.asarray(MFBmat[train_index:]).astype('float32')   
    y_train = np.asarray(bin_labels[:train_index]).astype('float32')
    y_test = np.asarray(bin_labels[train_index:]).astype('float32')
    min_xtrain = np.min(x_train_orig)
    x_test_abs = np.abs(x_test_orig / min_xtrain) 

#########-----data_augmentation to x train-----#########
    if augmentation == True :
        
        x_train_for_aug = MFBmat[:train_index]
        x_train_for_aug = expand_dims(x_train_for_aug, 3)
        x_train_for_aug = expand_dims(x_train_for_aug, 1)

########-------set data for bulbul net for augmentation

        X_train_Aug = data_augmentation(x_train = x_train_for_aug)
        X = np.vstack((x_train_for_aug, X_train_Aug, x_train_tp_aug))
        X_abs = np.abs(X[:,0,:,:,0]/min_xtrain )
        Y =  np.hstack((y_train, y_train , y_train))            
        
        shapeMat = ( x_train_orig.shape[1], x_train_orig.shape[2], 1)
        with open("data/train/data_50_1sec_full.pickle", 'wb') as f:
            pickle.dump([shapeMat , X_abs , Y , x_test_abs , y_test], f)
            
########-------set data for bulbul net for no augmentation
    
    else:
        X_train = expand_dims(x_train_orig, 3)
        X_train = expand_dims(X_train, 1)
        X = X_train
        X_abs = np.abs(X[:,0,:,:,0]/min_xtrain)
        Y = y_train
        
    shapeMat = ( x_train_orig.shape[1], x_train_orig.shape[2], 1)

#######-------run net
    if run_net == 'convnet':
        model, test_acc , TPR , FPR , FP_TP, TP, FP =BM.convnet(shape_mat = shapeMat , X = X_abs , Y = Y , x_test = x_test_abs , y_test = y_test ,
                  num_val = 0.1 , num_epochs = 100 , batchSize = 1024, n_kernels = 32 , Th_pred = 0.5)
    elif run_net == 'convnet_no_batch_n':
        model, test_acc , TPR , FPR , FP_TP, TP, FP = BM.convnet_no_batch_n(shape_mat = shapeMat , X = X_abs , Y = Y , x_test = x_test_abs , y_test = y_test ,
                  num_val = 0.1 , num_epochs = 100 , batchSize = 1024, n_kernels = 32 , Th_pred = 0.5)
    elif run_net == 'resnet':
        model, test_acc , TPR , FPR , FP_TP, TP, FP = BM.resnet(shape_mat = shapeMat , X = X_abs , Y = Y , x_test = x_test_abs , y_test = y_test ,
                  num_val = 0.1 , num_epochs = 100 , batchSize = 1024, n_kernels = 32, dropout = 0.3 , hidden_units = 90 , Th_pred = 0.5)
    elif run_net == 'resnet2':
        model, test_acc , TPR , FPR , FP_TP, TP, FP = BM.resnet2(shape_mat = shapeMat , X = X_abs , Y = Y , x_test = x_test_abs , y_test = y_test,
                  num_val = 0.1 , num_epochs = 100 , batchSize = 1024, n_kernels = 32 , Th_pred = 0.5)
    elif run_net == 'mini_xception':
        model, test_acc , TPR , FPR , FP_TP, TP, FP = BM.mini_xception(shape_mat = shapeMat , X = X_abs , Y = Y , x_test = x_test_abs , y_test = y_test ,
                  num_val = 0.1 , num_epochs = 100 , batchSize = 1024, n_kernels = 32 , Th_pred = 0.5)


    date = time.strftime("%Y-%m-%d")
    model.save('data/model_' + str(run_net)+date)
    return test_acc , TPR , FPR , FP_TP, TP, FP
