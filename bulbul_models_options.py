# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:08:44 2021

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

import Utils as Ut

def read_wav_label(frame_length , j , hop_length , labels ,L , Th , bin_l = 0):
    ntime_start = j * hop_length
    ntime_end = ntime_start + frame_length-1
    L = len(labels)
    for k in range(L): # search on all labels
        # case 1 - all label within segment
        if (labels[k][0] > ntime_start) and (labels[k][1] < ntime_end):
            bin_l = 1
        # case 2 - start of next (k+1) label and end of current (k) label
        # within segment (and k < L-1)
        if k < L-1 and (labels[k+1][0] < ntime_end) and (labels[k][1] > ntime_start):
            
            if ((ntime_end - labels[k+1][0]  ) + (labels[k][1] - ntime_start)) > Th * frame_length :
                bin_l = 1
        # case 3 - start within segment and end outside segment (from right)
        if (labels[k][0] < ntime_end) and (labels[k][0] > ntime_start)  and (labels[k][1] > ntime_end):
            if (ntime_end - labels[k][0]) > Th * frame_length :
                bin_l = 1
        # case 4 - end within segment and start outside (from left)
        if (labels[k][1] < ntime_end) and (labels[k][1] > ntime_start)  and (labels[k][0] < ntime_start):
            if (labels[k][1]- ntime_start ) > Th * frame_length :
                bin_l = 1
        # case 5 - all segment within label delimiters
        if (labels[k][0] < ntime_start) and (labels[k][1] > ntime_end):
            bin_l = 1
    return bin_l

def fadeout(audio_input, fs = 44100, duration=0.05):
     # convert to audio indices (samples)
     length = int(duration*fs)
     end = audio_input.shape[0]
     start = end - length
     fade_curve = np.linspace(1.0, 0.0, length)
 
     # apply the curve
     audio_input[start:end] = audio_input[start:end] * fade_curve
     return audio_input
 
def fadein(audio_input, fs = 44100, duration=0.05):
     # convert to audio indices (samples)
     length = int(duration*fs)
     start = 0
     end = length
     fade_curve = np.linspace(0.0, 1.0, length)
 
     # apply the curve
     audio_input[start:end] = audio_input[start:end] * fade_curve
     return audio_input  

def play_file(y , fs):
    y1 = y.copy()
    y2 = fadein(y1)
    y3 = fadeout(y2)
    sd.play(y3, fs)
    time.sleep(y.size/fs)
    
    return

def display_image(MFB_dB , bin_l , j , q , fs , hop_length , fmin , fmax , prdct = None):
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(MFB_dB, x_axis='time',
             y_axis='mel', sr=fs, hop_length = hop_length,
             fmin = fmin, fmax=fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
    ax.set(title='label = '+str(bin_l) + '  predict ='+str(prdct))
    
    plt.savefig('spectrograms/spec_num'+str(j)+'_'+str(q)+'.png')
    
    return


def gen_labels(min_xtrain = -80, data_path = 'data/to_label', 
               fs = 44100, dur = 0.5,  nfft = 2048, hop = 700,
               n_filters = 50, f_min = 1000,
               f_max = 3500, fL = 700, fH = 3900 , Th_pred = 0.5):    
    """
    gen_labels - generates label for candidate acoustic events
    detected by a given machine learning system.
    Input arguments: parameters for audio analysis, and path
    for directory with unlabeled audio files. 
    Returns: large_df - to generate text.    
    
    """
    labelsPath = data_path
    model = keras.models.load_model('data/model_convnet2021-09-25')
    # Set the frame parameters to be equivalent to the librosa defaults
    # in the file's native sampling rate
    frame_length = int(dur*fs) # (2048 * fs) // fs
    hop_length = int(frame_length/2) #(1024 * fs) // fs
    
    # get  list of all files  within labelsPath
    wav_files=glob.glob(labelsPath + "/*.wav", recursive=True)
    N = len(wav_files)
    
    large_df = pd.DataFrame([])
    
    for q in range(N):
         
        # Stream the data, working on 128 frames at a time
        fs = librosa.get_samplerate(str(wav_files[q]))
        stream = librosa.stream(str(wav_files[q]),
                                block_length=1,
                                frame_length=frame_length,
                                hop_length=hop_length)        
        data_files = []
        ntime_start = 0
        j = 0
        for y in stream:
            if y.size == frame_length:
                   
               ##calculate mel spectogram
               MFB = librosa.feature.melspectrogram(y , sr = fs, 
                         n_fft = nfft,
                         hop_length = hop, n_mels=n_filters,       
                         fmax=f_max, fmin = f_min)
               # MFB_dB = librosa.power_to_db(MFB, ref=np.max)
               MFB_med = Ut.medclip(MFB, 3.5, 1e-14)
               MFB_dB1 = librosa.power_to_db(MFB_med, ref=np.max)
               MFB_dB = Ut.blobRemove(MFB_dB1, 3, floordB = -80, Thresh = -60 )
              
                
               ## predict -
               X_test_1 = MFB_dB.reshape(1, MFB_dB.shape[0], MFB_dB.shape[1])
               X_test_1 = X_test_1 / min_xtrain
               y_prdct = model.predict(X_test_1)
               prdct = (np.sign(y_prdct - Th_pred) + 1) / 2
             
               data_files.append([q , j, MFB_dB, prdct])
               column = ['index_dir', 'index_file', 'MFB', 'prediction']
               j = j + 1
               if np.mod(j,100) == 0:
                   print(j)
           
    
        df = pd.DataFrame(data_files , columns = column)
    
        large_df = pd.concat([large_df,df], ignore_index=True)
    return large_df 


def gen_label_txt(label_df, frame_length = 0.5, hop_length = 0.25):
    """
    gen_label_txt - generates label text for candidate acoustic events
    detected by a given machine learning system to use with Audacity
    Input arguments: label_df - data frame generated by gen_labels
    frame_length - duration (seconds) of each segment, hop_length - hop size 
    Returns: label_time - start and end of the candidate events

    """
    prdction = label_df['prediction'].tolist() # read series, convert to list
    # prediction - the results of trained NN. 
    prdct = np.array(prdction)  
    diff_prdct = (prdct[1:]-prdct[:-1]) # 1 indicates start, -1 - end of detected event
    diff_prdct = diff_prdct.reshape(diff_prdct.size)
    start = np.where(diff_prdct == 1) #indices of starts
    start = np.array(start)
    start = start.reshape(start.size,1) # indices of ends
    start_sec = start * hop_length + hop_length # try setting start point
    # start_sec = start_sec - frame_length # widening each event to 
    # include one more frame_length from left
    
    stop = np.where(diff_prdct == - 1) #  indices of ends of detected events
    stop = np.array(stop)
    stop = stop.reshape(stop.size,1)
    stop_sec = stop * hop_length + frame_length # ending point 
    # stop_sec = stop_sec + 3*hop_length #widening to the rigth.
    label_1 = np.ones([start_sec.size,1])
    if start_sec.size == stop_sec.size :
        label_sec = np.concatenate([start_sec, stop_sec, label_1], axis = 1)
        ## if dimention dont fit:
    elif start_sec.size > stop_sec.size :
        label_sec = np.concatenate([start_sec[:-1], stop_sec, label_1[:-1]], axis = 1)
    elif start_sec.size < stop_sec.size :
        label_sec = np.concatenate([start_sec, stop_sec[1:], label_1[1:]], axis = 1)

    label_time = tuple(map(tuple, label_sec)) # converting to a list of tupples
    
    # generating the candidate
    date = time.strftime("%Y-%m-%d")
    fname = 'label_' + date + '.txt'
    #destpath = os.path.join(dest_dir, fname)
    f = open(fname, "w+")
    for ts in label_time:
        f.write('{:.6f}\t{:.6f}\t{}\n'.format(ts[0], ts[1], int(ts[2])))
    f.close()
    
    return label_time




def run_data(data_path  , min_xtrain = -80.0 , 
               fs = 44100 , dur = 0.5 ,  nfft = 2048 , hop = 700,
               n_filters = 50 , f_min = 1000 , f_max = 3500):

    model = keras.models.load_model('data/model_convnet2021-09-25')
    # Set the frame parameters to be equivalent to the librosa defaults
    # in the file's native sampling rate
    frame_length = int(dur*fs) # (2048 * fs) // fs
    hop_length = int(frame_length/2) #(1024 * fs) // fs   
    
    ### a process to all directory by locations - - 
    # get  list of all files  within labelsPath
    wav_files=glob.glob(data_path + "/*/*.wav", recursive=True)
    # replace "\\" with "/"
    wav_files=[re.sub("\\\\", "/", x) for x in wav_files]
    # extract location folder and text file name from each file path
    regexp=re.compile(data_path+"(\/([^\/]+).*\/([^\/]+\.wav))")
    files_tuples = [regexp.search(x).group(1, 2, 3)   for x in wav_files] 
    
    N = len(wav_files)
    W = 0
    
    ##lists for dataframe:   
    df_pred = pd.DataFrame([])
    dur_list = []
    word_edge = []
    
    # for q in range(N):
    for q in files_tuples:
        
        path = q[0]
        location = q[1]
        filename = q[2]
        match = re.search(r'S4A[0-9]{5}_([0-9]{8})_([0-9]{6})', filename)
        date = str(match.group(1))
        time = str(match.group(2))
        # Stream the data, working on 128 frames at a time
        fs = librosa.get_samplerate(str(wav_files[W]))
        stream = librosa.stream(str(wav_files[W]),
                                block_length=1,
                                frame_length=frame_length,
                                hop_length=hop_length)        
        data_files = []
        ntime_start = 0
        j = 0
        for y in stream:
            if y.size == frame_length:
              
               ##calculate mel spectogram
               MFB = librosa.feature.melspectrogram(y , sr = fs, 
                         n_fft = nfft,
                         hop_length = hop, n_mels=n_filters,       
                         fmax=f_max, fmin = f_min)
               # MFB_dB = librosa.power_to_db(MFB, ref=np.max)
               MFB_med = Ut.medclip(MFB, 3.5, 1e-14)
               MFB_dB1 = librosa.power_to_db(MFB_med, ref=np.max)
               MFB_dB = Ut.blobRemove(MFB_dB1, 3, floordB = -80, Thresh = -60 )
               
               ## predict -
               X_pred = MFB_dB.reshape(1, MFB_dB.shape[0], MFB_dB.shape[1])
               X_pred = X_pred / min_xtrain
               prdct = np.round(model.predict(X_pred))
               prdct = prdct.item()
               
               # time_start = j*0.025 
               # time_end = time_start + 0.5    
               
               data_files.append([filename , location , date , time ,  j , MFB_dB , prdct]) #time_start , time_end])
               column = ['file_name' , 'location' , 'date' , 'time' ,'index_file' , 'MFB', 'prediction'] # 'time_start' , 'time_end']
               j = j + 1
               if np.mod(j,100) == 0:
                    print(j)
                

        duration = j * (dur/2)
        df = pd.DataFrame(data_files , columns = column) 
        
        ######## creat a list of words edges for every file:
        bin_pred = df['prediction'].tolist()
        prdct = np.array(bin_pred)  
        diff_prdct = (prdct[1:]-prdct[:-1]) # 1 indicates start, -1 - end of detected event
        diff_prdct = diff_prdct.reshape(diff_prdct.size)
        start = np.where(np.isin(diff_prdct, 1))[0].tolist() #indices of starts
        start_sec = [i * hop_length for i in start]  # include one more frame_length from left
    
        stop = np.where(np.isin(diff_prdct, -1))[0].tolist() #  indices of ends of detected events
        stop_sec = [i * hop_length + frame_length for i in stop] # ending point     
    
        word_zip = list(zip(start_sec, stop_sec))
        word_edge.append(word_zip)
        
        df_pred = pd.concat([df_pred ,df], ignore_index=True)
        dur_list.append(duration) #time in seconds

        W += 1
        print('W =', W)
        
    df_pred.to_pickle('data/df_pred'+'.pkl')

    return df_pred , dur_list , word_edge, files_tuples





def count_calls(files , words , dur_list):
 
    """
    Input arguments:
        files - files_tuples - list of tuples of all files from all directories
        path, location and filename for each
        Words - word_edge -list of tuples of start and end
        dur_list - list of duration for each file
        path - location of directory
    Returns:
       count_file - dataframe of file_name , lacation , date , time , duration, count_calls , calls per dur
       count_by_days - location , date , sum_calls , sum_duration , calls per day
    """
    count_file = pd.DataFrame([])
    count_list = []
    iter_file = 0
    for one_file in files:    #list of tuples of all files from all directory    
        pathfile = one_file[0]
        location = one_file[1]
        filename = one_file[2]
        file = filename[:-4] #without .wav
        time = file[18:]
        date = file[9:17]
        words_edge = words[iter_file]  #list of tuples of start time and end time of words detection
        general_dur = dur_list[iter_file]
        
        iter_file+=1
        iter_word = 0
        calls_dur = 0
        count_call = len(words_edge)
        for one_word in words_edge: #looping over one list of words from one file
            # Load x seconds of a file (calcuated by end and start), starting y seconds in (start time)
            t_start = (1/44100) * one_word[0]
            t_end = (1/44100) * one_word[1]
            dur = t_end - t_start
            calls_dur += dur
            iter_word += 1
    
        count_list.append([pathfile, location ,filename , date , time , general_dur , calls_dur , count_call])
        column = ['path' , 'location' , 'file_name' , 'date' , 'time' ,'general_dur' , 'calls_dur', 'count_call']
        
        
    df = pd.DataFrame(count_list , columns = column) 
    count_file = pd.concat([count_file ,df], ignore_index=True)
  ###count calls per file-     
    count_file['calls per hour'] = count_file['count_call']/count_file['general_dur']*3600
    count_file['ratio calls per dur'] = count_file['calls_dur']/count_file['general_dur']
  ###according to date- creat dataframe of date and amount of calls to this date
    count_by_days1 = count_file.groupby(['location' , 'date']).agg({'count_call':['sum']})
    count_by_days1.columns = count_by_days1.columns.droplevel(0)
    count_by_days1 = count_by_days1.reset_index()
    count_by_days1 = count_by_days1.rename(columns={'sum':'sum_calls'}) 
    
    count_by_days2 = count_file.groupby(['location' , 'date']).agg({'general_dur':['sum']})
    count_by_days2.columns = count_by_days2.columns.droplevel(0)
    count_by_days2 = count_by_days2.reset_index()
    count_by_days2 = count_by_days2.rename(columns={'sum':'sum_duration'})
    count_by_days = pd.merge(count_by_days1, count_by_days2, on=['location','date'])

    count_by_days['calls per day'] = count_by_days['sum_calls']/count_by_days['sum_duration']*3600


#####plot how many calls per day for each location-
    # x- days y-calls
    locations = count_by_days['location'].unique().tolist()
    j=1
    for l in locations:
        df_loc = count_by_days.loc[count_by_days.location == l]
        plt.figure(j, figsize=(14, 10))
        count_plot = sns.barplot(x = 'date', y = "calls per day" , data = df_loc , dodge = False )        
        plt.xticks(rotation=90,fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('frequency of bulbul calls over time in - ' + str(l),fontsize=15)
        plt.ylabel('amount of calls - measure of calls/duration',fontsize=15)

        plt.show(count_plot)
        j+=1
        
    return count_file , count_by_days


def cut_edges(word, fs , fL, fH, frame_length, hop_size, prcntl = 20,
              Th2 = 0.1, sigflag = 1):
    """
    cut_edges - trim the edges of an audio sound recording 
    (Bird call) so that the "word" will start and finish without low energy
    background noise before and after the call.
    cut_edges uses two threshold (a permissive and conservative) to demarcate
    the word.
    Input arguments: word - a path for the wav file of the word to be trimmed.
    fL, fH - low and high frequency for bandpass filter
    frame_length, hop_size - for obtaining the signal short-time (bandpassed)
    energy.
    prctl - percentile for the threholds to be used for finding the edges.
    Th2 - a treshold for using as permissive or conservative threshold.
    Returns - x_cut - a numpy array containing the trimmed word.
    
    """
    if sigflag == 0:
        x, sr = librosa.load(word, sr = None)
        fs = sr
    else:
        x = word
        
    sos = signal.butter(35, np.array([fL,fH]), 'bp', fs = fs, output='sos')
    x = signal.sosfilt(sos, x)
    yorig, Eorig = Ut.Energy(x, fs, frame_length, hop_size)
    #E = np.log10(E)
    E = (Eorig - np.min(Eorig)) / (np.max(Eorig) - np.min(Eorig))
    y = (yorig - np.min(yorig)) / (np.max(yorig) - np.min(yorig))
    pr = np.percentile(E, prcntl)
    Thpr = pr + 0.02
    #Thpr = 2*np.mean(E[E<pr])
    ThL = min(Thpr, Th2)
    ThH = max(Thpr, Th2)
    Lx = x.size
    Ly = y.size
    
    start = 0
    stop = Ly
    for k in range(Ly): # left side forward
        if y[k] < ThH:
            start = k + 1
        else:
            break
    for j in range(Ly - k): # left side backward
        if y[k-j] > ThL:
            start = start - 1
        else:
            break
        
        
    for k in range(Ly): # right side backword
        if y[Ly-1-k] < ThH:
            stop = Ly - k - 1
        else:
            break
    for j in range(k): # right side forword
        if y[min(Ly-k+j,Ly-1)] > ThL:
            stop = Ly-k+j
        else:
            break
        
    start = max(0, start)
    stop = min(stop, Lx - 1)
    x_cut = x[start:stop]
    ThvecL = ThL * np.ones([Ly,])
    ThvecH = ThH * np.ones([Ly,])
    tt = np.arange(0, Ly/fs, 1/fs)
    tt = tt[:Ly]
    
    #plot word edges-
    # plt.plot(tt, y, tt, ThvecL,'r--',tt, ThvecH,'g--', 
    #          tt[start], y[start], 'mo', tt[stop], y[stop], 'mo',
    #          tt[:Lx], 0.1*x-0.2, 'c.-')
    # plt.show()
    # plt.pause
    
    return x_cut
    
def cut_words(files , words , path): 
    """
    Input arguments:
        files - list of tuples of all files from all directories
        path, location and filename for each
        Words - list of tuples of start and end
        path - location of directory
    Returns:
        cut_words save all cut words in a directory.
        
    """
    fL = 1000
    fH = 3500
    frame_length = 1024
    hop_size = 512    
    iter_file = 0
    for q in files:    #list of tuples of all files from all directory    
        pathfile = q[0]
        location = q[1]
        filename = q[2]
        file = filename[:-4] #without .wav
        words_edge = words[iter_file]  #list of tuples of start time and end time of words detection
        
        iter_file+=1
        iter_word = 0
        readfile = (path+pathfile)
        # Creat directory or check whether the specified path exists
        directory = location
        parent_dir = 'data/saved_words/'
        path_dir = os.path.join(parent_dir, directory)
        
        #Check-
        isExist = os.path.exists(path_dir)
        if not isExist:
            # Create a new directory
            os.makedirs(path_dir)
            print("Directory '% s' created" % directory)
        
        
        for one_word in words_edge: #looping over one list of words from one file
            # Load x seconds of a file (calcuated by end and start), starting y seconds in (start time)
            t_start = (1/44100) * one_word[0]
            t_end = (1/44100) * one_word[1]
            dur = t_end - t_start
            y, sr = librosa.load(readfile, sr=44100 , offset = t_start , duration=dur)
            word_cut = cut_edges(y, sr, fL, fH, frame_length, hop_size, prcntl = 20, Th2 = 0.05 , sigflag = 1)
            # sf.write('data/saved_words/'+location+'/'+file+'_word_'+str(iter_word)+'.wav', word_cut, sr)
            sf.write(path_dir+'/'+file+'_word_'+str(iter_word)+'.wav', word_cut, sr)
            iter_word += 1
  
    return


def model_pred(model, x_test, y_test , Th_pred):
    y_test_v = y_test.reshape(y_test.shape[0],1)
    y_test_pred = model.predict(x_test)
    y_test_pred_bin = (np.sign(y_test_pred - Th_pred) +1)/2
    test_diff = (np.abs(y_test_pred_bin - y_test_v))
    test_acc = 1 - np.sum(test_diff)/ y_test.size
    FP = np.sum(((y_test_v - y_test_pred_bin) == -1))
    TP = float(np.dot(y_test_v.T, y_test_pred_bin))
    FPR = FP / np.sum(y_test_v == 1)
    TPR = TP/np.sum(y_test_v == 1)
    FP_TP = FP/TP

    print('test accuracy =', test_acc)
    print('True positive rate = ', TPR)
    print('False positive rate = ', FPR)
    print('FP/TP = ', FP_TP)
    
    return test_acc , TPR , FPR , FP_TP, TP, FP

def draw_graphs(epochs , loss_values , val_loss_values , acc_values , val_acc_values):
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    date = time.strftime("%Y-%m-%d")
    plt.savefig(date +'loss.png')

    plt.clf
    plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
    plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()
    plt.savefig(date +'accuracy.png')
    
    return


def convnet(shape_mat ,X , Y , x_test , y_test ,  num_val = 0.1 ,
             num_epochs = 1 , batchSize = 64, n_kernels = 32, dropout = 0.5, 
             hidden_units = 90 , Th_pred = 0.5):
    inputs = keras.Input(shape=shape_mat)
    #x = data_augmentation(inputs)
    #x = layers.experimental.preprocessing.Rescaling(np.abs(1./min_xtrain))(x)
    #x = layers.Conv2D(filters=32, kernel_size=(3), activation="relu", padding = 'same')(inputs)
    x = layers.Conv2D(n_kernels, 3, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=n_kernels*2, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=n_kernels*4, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, padding = 'same')(x)
    x = layers.Conv2D(filters=n_kernels*8, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, padding = 'same')(x)
    x = layers.Conv2D(filters=n_kernels*8, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_units, activation = "relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    #model.summary()
    
    # using callbacks to save the best performing model
    # callbacks = [
    # keras.callbacks.ModelCheckpoint("bulbul_conv.keras",
    #                                 save_best_only=True)
    # ]
    # compiling the model
    model.compile(optimizer = 'rmsprop', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

    # Validation - monitor during training the accuracy of the model 
    # on data it has never seen before - creation of validation set
    
    val_index = int(num_val*X.shape[0]) 
    x_val = X[:val_index]
    partial_x_train = X[val_index:]
    y_val = Y[:val_index]
    partial_y_train = Y[val_index:]

    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = num_epochs,
                    batch_size = batchSize,
                    validation_data = (x_val, y_val))



#    model = keras.models.load_model("bulbul_conv.keras")
    # print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
    results = model.evaluate(x_test, y_test)

    history_dict = history.history
    history_dict.keys()
    [u'acc', u'loss', u'val_acc', u'val_loss']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    epochs = range(1, len(acc_values) + 1)
    
    #draw training and validation loss and accuracy:
    draw_graphs(epochs , loss_values , val_loss_values , acc_values , val_acc_values )

    test_acc , TPR , FPR , FP_TP, TP, FP = model_pred(model , x_test , y_test , Th_pred)
    #model.save('data')
    
    return model, test_acc , TPR , FPR , FP_TP, TP, FP

def convnet_no_batch_n(shape_mat ,X , Y , x_test , y_test ,  num_val = 0.1 ,
             num_epochs = 1 , batchSize = 64, n_kernels = 32, dropout = 0.5, 
             hidden_units = 90 , Th_pred = 0.5):
    inputs = keras.Input(shape=shape_mat)
    #x = data_augmentation(inputs)
    #x = layers.experimental.preprocessing.Rescaling(np.abs(1./min_xtrain))(x)
    #x = layers.Conv2D(filters=32, kernel_size=(3), activation="relu", padding = 'same')(inputs)
    x = layers.Conv2D(n_kernels, kernel_size=3, activation="relu", padding = 'same')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=n_kernels*2, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=n_kernels*4, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.MaxPooling2D(pool_size=2, padding = 'same')(x)
    x = layers.Conv2D(filters=n_kernels*8, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.MaxPooling2D(pool_size=2, padding = 'same')(x)
    x = layers.Conv2D(filters=n_kernels*8, kernel_size=3, activation="relu", padding = 'same')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_units, activation = "relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    # using callbacks to save the best performing model
    # callbacks = [
    # keras.callbacks.ModelCheckpoint("bulbul_conv.keras",
    #                                 save_best_only=True)
    # ]
    # compiling the model
    model.compile(optimizer = 'rmsprop', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

    # Validation - monitor during training the accuracy of the model 
    # on data it has never seen before - creation of validation set
    
    val_index = int(num_val*X.shape[0]) 
    x_val = X[:val_index]
    partial_x_train = X[val_index:]
    y_val = Y[:val_index]
    partial_y_train = Y[val_index:]

    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = num_epochs,
                    batch_size = batchSize,
                    validation_data = (x_val, y_val))



#    model = keras.models.load_model("bulbul_conv.keras")
    # print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
    results = model.evaluate(x_test, y_test)

    history_dict = history.history
    history_dict.keys()
    [u'acc', u'loss', u'val_acc', u'val_loss']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    epochs = range(1, len(acc_values) + 1)

    #draw training and validation loss and accuracy:
    draw_graphs(epochs , loss_values , val_loss_values , acc_values , val_acc_values )

    test_acc , TPR , FPR , FP_TP, TP, FP = model_pred(model , x_test , y_test , Th_pred)
    #model.save('data')
    
    return model, test_acc , TPR , FPR , FP_TP, TP, FP
        
def resnet(shape_mat ,X , Y , x_test , y_test ,  num_val = 0.1,
           num_epochs = 1 , batchSize = 1024, n_kernels=32,
           dropout = 0.5 , hidden_units = 90 , Th_pred = 0.5):
    inputs = keras.Input(shape=shape_mat)
    #x = data_augmentation(inputs)
    #x = layers.experimental.preprocessing.Rescaling(np.abs(1./min_xtrain))(x)
    #x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
    
    def residual_block1(x, filters , filtsize1, filtsize2, pooling=False):
        # Regular resnet block
        residual = x
        x = layers.Conv2D(filters, filtsize1, activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, filtsize2, activation="relu", padding="same")(x)
        if pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2)(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.add([x, residual])
        return x
    
    def residual_block2(x, filters, filtsize1, filtsize2, pooling=False):
        #  Resnet with Batch normalization
        residual = x
        x = layers.Conv2D(filters, filtsize1, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, filtsize2, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        if pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2)(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.add([x, residual])
        return x
    
    def residual_block3(x, filters, filtsize1, filtsize2, pooling=False):
        #  Resnet with Batch normalization before applying activation
        residual = x
        x = layers.Conv2D(filters, filtsize1, use_bias = False, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, filtsize2,use_bias = False, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x) 
        if pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2)(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.add([x, residual])
        return x

    x = residual_block2(inputs, filters=n_kernels, filtsize1 = 5 ,filtsize2=3,pooling=True)
    x = residual_block2(x, filters=n_kernels, filtsize1 = 3 ,filtsize2=3, pooling=True)
    x = residual_block2(x, filters=n_kernels*2, filtsize1 = 3 ,filtsize2=3, pooling=True)
    x = residual_block2(x, filters=n_kernels*4, filtsize1 = 3 ,filtsize2=3, pooling=False)
    x = residual_block2(x, filters=n_kernels*4,filtsize1 = 3 ,filtsize2=3, pooling=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(hidden_units, activation = "relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    # compiling the model
    model.compile(optimizer = 'rmsprop', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

    # Validation - monitor during training the accuracy of the model 
    # on data it has never seen before - creation of validation set
    
    val_index = int(num_val*X.shape[0]) 
    x_val = X[:val_index]
    partial_x_train = X[val_index:]
    y_val = Y[:val_index]
    partial_y_train = Y[val_index:]

    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = num_epochs,
                    batch_size = batchSize,
                    validation_data = (x_val, y_val))

    results = model.evaluate(x_test, y_test)

    history_dict = history.history
    history_dict.keys()
    [u'acc', u'loss', u'val_acc', u'val_loss']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    epochs = range(1, len(acc_values) + 1)

    #draw training and validation loss and accuracy:
    draw_graphs(epochs , loss_values , val_loss_values , acc_values , val_acc_values )

    test_acc , TPR , FPR , FP_TP, TP, FP = model_pred(model , x_test , y_test , Th_pred)
    #model.save('data/model_resnet2')
    return model, test_acc , TPR , FPR , FP_TP, TP, FP

def resnet2(shape_mat ,X , Y , x_test , y_test ,  num_val = 0.1,
           num_epochs = 1 , batchSize = 64, n_kernels=32 , Th_pred = 0.5):
    inputs = keras.Input(shape=shape_mat)
    #x = data_augmentation(inputs)
    #x = layers.experimental.preprocessing.Rescaling(np.abs(1./min_xtrain))(x)
    #x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
    
    def residual_block1(x, filters , filtsize1, filtsize2, pooling=False):
        # Regular resnet block
        residual = x
        x = layers.Conv2D(filters, filtsize1, activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, filtsize2, activation="relu", padding="same")(x)
        if pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2)(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.add([x, residual])
        return x
    
    def residual_block2(x, filters, filtsize1, filtsize2, pooling=False):
        #  Resnet with Batch normalization
        residual = x
        x = layers.Conv2D(filters, filtsize1, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, filtsize2, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        if pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2)(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.add([x, residual])
        return x
    
    def residual_block3(x, filters, filtsize1, filtsize2, pooling=False):
        #  Resnet with Batch normalization before applying activation
        residual = x
        x = layers.Conv2D(filters, filtsize1, use_bias = False, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, filtsize2,use_bias = False, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x) 
        if pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2)(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.add([x, residual])
        return x

    x = residual_block2(inputs, filters=n_kernels, filtsize1 = 5 ,filtsize2=3,pooling=True)
    x = residual_block2(x, filters=n_kernels, filtsize1 = 3 ,filtsize2=3, pooling=True)
    x = residual_block2(x, filters=n_kernels*2, filtsize1 = 3 ,filtsize2=3, pooling=True)
    x = residual_block2(x, filters=n_kernels*4, filtsize1 = 3 ,filtsize2=3, pooling=False)
    x = residual_block2(x, filters=n_kernels*4,filtsize1 = 3 ,filtsize2=3, pooling=False)
    x = residual_block2(x, filters=n_kernels*8, filtsize1 = 3 ,filtsize2=3, pooling=False)
    x = residual_block2(x, filters=n_kernels*8,filtsize1 = 3 ,filtsize2=3, pooling=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    # compiling the model
    model.compile(optimizer = 'rmsprop', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

    # Validation - monitor during training the accuracy of the model 
    # on data it has never seen before - creation of validation set
    
    val_index = int(num_val*X.shape[0]) 
    x_val = X[:val_index]
    partial_x_train = X[val_index:]
    y_val = Y[:val_index]
    partial_y_train = Y[val_index:]

    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = num_epochs,
                    batch_size = batchSize,
                    validation_data = (x_val, y_val))

    results = model.evaluate(x_test, y_test)

    history_dict = history.history
    history_dict.keys()
    [u'acc', u'loss', u'val_acc', u'val_loss']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    epochs = range(1, len(acc_values) + 1)

    #draw training and validation loss and accuracy:
    draw_graphs(epochs , loss_values , val_loss_values , acc_values , val_acc_values )

    test_acc , TPR , FPR , FP_TP, TP, FP = model_pred(model , x_test , y_test , Th_pred)
    #model.save('data/model_resnet2')
    return model, test_acc , TPR , FPR , FP_TP, TP, FP



    
    
def mini_xception(shape_mat ,X , Y , x_test , y_test ,  num_val = 0.1 ,
         num_epochs = 1 , batchSize = 64, n_kernels = 32, n_iter = 5, dropout = 0.5 , Th_pred = 0.5):
    inputs = keras.Input(shape=shape_mat)
    #x = data_augmentation(inputs)
    #x = layers.experimental.preprocessing.Rescaling(np.abs(1./min_xtrain))(x)
    #x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
 
    x = layers.Conv2D(filters=n_kernels, kernel_size=5, use_bias=False)(inputs)
    
    for size in range(n_iter) :
        residual = x
    
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(n_kernels*2**size, 3, padding="same", use_bias=False)(x)
    
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(n_kernels*2**size, 3, padding="same", use_bias=False)(x)
    
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
        residual = layers.Conv2D(
            n_kernels*2**size, 1, strides=2, padding="same", use_bias=False)(residual)
        x = layers.add([x, residual])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    # compiling the model
    model.compile(optimizer = 'rmsprop', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

    # Validation - monitor during training the accuracy of the model 
    # on data it has never seen before - creation of validation set
    
    val_index = int(num_val*X.shape[0]) 
    x_val = X[:val_index]
    partial_x_train = X[val_index:]
    y_val = Y[:val_index]
    partial_y_train = Y[val_index:]

    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = num_epochs,
                    batch_size = batchSize,
                    validation_data = (x_val, y_val))

    results = model.evaluate(x_test, y_test)

    history_dict = history.history
    history_dict.keys()
    [u'acc', u'loss', u'val_acc', u'val_loss']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    epochs = range(1, len(acc_values) + 1)

    #draw training and validation loss and accuracy:
    draw_graphs(epochs , loss_values , val_loss_values , acc_values , val_acc_values )

    test_acc , TPR , FPR , FP_TP, TP, FP = model_pred(model , x_test , y_test , Th_pred)
    # model.save('data/model_resnet2')
    return model, test_acc , TPR , FPR , FP_TP, TP, FP

