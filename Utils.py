# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:19:47 2021

@author: AM and YL
""" 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

    
def f0_synth(sig, siglen,f0, fs, frame_length):
    """     
    Parameters: x - original signal, siglen - length of the audio signal. 
    f0 - np array of pitch
    contour. fs - sampling frequency. frame_length - length of each frame
    for sinusoidal signal synthesis.
    ----------
    returns - y - a numpy array signal which composed of concateneation of
    sinusoidal signals, one for each frame, with continous phase in the boundaries
    between two consecutive sinusoids.

    """
    # import numpy as np
    # from scipy.signal import hilbert
    
    j=0
    phi = 0
    y = np.zeros(siglen)
    Ts = 1 / fs
    tt = np.linspace(0, float(frame_length/fs), frame_length)
    N = tt.size
    for i in range(0, siglen, frame_length):
        if j >= f0.size:
            break
        freq = f0[j]
        T0 = 1 / freq
        framesin=np.cos(2*np.pi*freq*tt+phi)
        y[i:i+frame_length] = framesin
        if framesin[-1] < framesin[-2]:
            phi = np.arccos(framesin[-1])+ Ts/T0*2*np.pi
        else:
            phi = 2*np.pi-np.arccos(framesin[-1])+ Ts/T0*2*np.pi          
        j = j + 1
        
    analytic_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    # plt.plot(amplitude_envelope)
    # plt.show()
    y = y*amplitude_envelope[0:y.size]
    
    return y


def Energy(sig, fs, frame_length, hop_size):
    """
    
    """
    import numpy as np
    import sys, os
    import matplotlib.pyplot as plt
    import re 
    from scipy import signal
    from scipy.signal import hilbert
    
    siglen = sig.size
    print(siglen)
    N = int(np.ceil(siglen/hop_size))
    E=np.zeros(N)    
    j=0
    y = np.zeros(int(siglen))    
    for i in range(0, siglen, hop_size):
        if i>=siglen:
            break
        E[j] = 1/frame_length*np.sum(sig[i:i+frame_length] ** 2)
        y[i:min(i+frame_length, siglen)] = E[j]
        j = j + 1
    E = smooth(E, window_len=5 )
    y = smooth(y, window_len = hop_size*2)
    y = y[:siglen]
    return y, E


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def nextpow2(m):
    """
    Calculating the next power of 2 for a given number m
    """
    y = int(2**(np.ceil(np.log2(m))))
    return(y)

def medclip(X, fctr, floordB = 0):
    """
    medclip carried out a median clipping for a given input matrix X.
    The function assigned a minimun predefined value for entries with values
    smaller or equal to a factor multiplied by the max of row and column medians
    of each entry.
    Input arguments:
    ----------------
    X - input matrix, fctr - multiplication factor, floordB - if a dB scale is
        required.    
    Returns:
    --------
    Y - the matrix after median clipping.
    """
    Xmedcols = np.median(X,0) 
    Xmedrows = np.median(X,1)
    Y = np.zeros(X.shape) + floordB
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > fctr*max(Xmedcols[j], Xmedrows[i]):
                Y[i,j] = X[i,j]
    return Y

def blobRemove(M,neigh_num = 2, floordB = -80, Thresh = -60 ):
    """
    Removing of isolated cells in a spectrogram (or mel-spectrogram) based on
    the number of neighbors of each cell. The spectrogram could be in a dB or a 
    linear scale. Please note that for the latter the default parameters should
    be modified.
    Input  arguments: 
    ----------------- 
        M - original spectrogram (image), neigh_num - number of neighbors, 
        floordB - value if number of neighbors<neigh_num. The default is -80.
        Thresh - threshold above which  1's are assigned in a binary matrix.
            The default is -60. 
    Returns
    -------
    M : the original matrix after removal of isolated cells (entries).
    """
    X = np.zeros(M.shape)
    X[M>Thresh] = 1
    Mup = X[0:-2,1:-1]
    Mdown = X[2:,1:-1]
    Mleft = X[1:-1,0:-2]
    Mright = X[1:-1,2:]
    Mul = X[0:-2,0:-2]
    Mur = X[0:-2, 2:]
    Mdl = X[2:, 0:-2]
    Mdr = X[2:, 2:]
    Mneigh = Mup + Mdown + Mleft + Mright + Mul + Mur + Mdl + Mdr
    M1 = M[1:-1,1:-1]
    M1[Mneigh<neigh_num] = floordB
    M[1:-1,1:-1] = M1
    return M
    
