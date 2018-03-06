import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import Conversion as convert
import scipy
from scipy import fftpack
#import itertools as itt
#from scipy import signal
#import normalize_easy as ne
#import matplotlib as mp
#from scipy import ndimage
# =============================================================================

def output_features(filepath, win_size, hop_size, min_freq, max_freq, num_mel_filts, n_dct, num_diags, len_delWin):
# Compute feature vector of the below method
# =============================================================================  
    mfccs, t = compute_mfccs(filepath, win_size, hop_size, min_freq, max_freq, num_mel_filts, n_dct)
    
    #Create window for delta & double-delta
    dWin = np.linspace(1, len_delWin, num = len_delWin)
    
    #Compute delta & double-delta coefficients
    delta = delta_coeffs(mfccs, dWin) 
    deltadelta = delta_coeffs(delta, dWin) 
    featureMatrix= np.concatenate((mfccs, delta, deltadelta), axis = 1)
    means = np.average(featureMatrix, axis = 0 )
    covMat = np.cov(featureMatrix, rowvar = False) 
    covDiags = getDiagonals(covMat, num_diags)
    
    return np.concatenate((means, covDiags), axis = 0);

def compute_mfccs(filepath, win_size, hop_size, min_freq, max_freq, num_mel_filts, n_dct):
# Compute MFCCs, delta, and doubledelta mean and covariance from audio file.
# ============================================================================
    fs, x_t = scipy.io.wavfile.read(filepath) #Read in .wav file
    x_t = x_t/np.max(abs(x_t)) #Normalize audio file to [-1, 1]
    
    #Handle zero padding extensions
    delIdx = []
    for i in range (0, len(x_t)):
        if (x_t[i] == 0):
            delIdx.append(i) 
    
    x_t = np.delete(x_t, delIdx)
        
    overlap = win_size-hop_size #Compute overlap
    windowF = np.hamming(win_size) #Create fft window
    s, f, t, image = plt.specgram(x_t, Fs = fs, window = windowF, NFFT = win_size, noverlap = overlap, mode = 'magnitude') #compute spectrogram
    
    min_mel = convert.hz2mel(min_freq) #Get minimum value of filterbank in mel 
    max_mel = convert.hz2mel(max_freq) #Get maximum value of filterbank in mel 
    mel_c = np.linspace(min_mel,max_mel,num=num_mel_filts+2) #linear spaced filters in mel domain
    freq_c = convert.mel2hz(mel_c) #mel centers back to Hz
    
    #Rounding to closest bin
    bin_c = convert.find_nearest(f, freq_c) 
    
    #Instantiate filterbank arrays
    bank = np.zeros((np.size(f),num_mel_filts)) 
    bank_norm = np.zeros((np.size(f),num_mel_filts))

    #Constructing mel filterbank
    for i in range(0,num_mel_filts):
        pre = np.zeros(((bin_c[i])), dtype=np.int)
        up = np.linspace(0,1,num=(bin_c[i+1]-bin_c[i])+1)
        down = np.linspace(1,0,num=(bin_c[i+2]-bin_c[i+1])+1)
        down = down[1:]
        post = np.zeros(((np.size(f)-bin_c[i+2])-1), dtype=np.int)
        bank[:,i] = np.concatenate([pre,up,down,post])
        bank_norm[:,i] = np.divide(bank[:,i],(np.sum(bank[:,i]))) #normalize to unit sum
        #plt(bank_norm)
    
    #Log magnitude & multiplication with filterbank
    logmag = 20*np.log10(np.dot(np.transpose(s),(bank_norm)))
    
    #Cosine Transform
    dctsig = fftpack.dct(logmag, norm='ortho')  

    #Removing 1st coefficient
    dctsig = np.delete(dctsig, 0, axis=1)
    del_col = np.arange(n_dct-1,num_mel_filts-1)
    mfccs = np.delete(dctsig, del_col, axis=1)

    
    return mfccs, t

def delta_coeffs(features, dWin):
#Compute delta and double-delta coefficients
# ============================================================================    
    normDelta = 2 * np.sum(np.square(dWin)) #get delta norm constant
    deltaArray = np.zeros(features.shape)
    
    #Create delta/double-delta coefficient array
    for i in range(0, features.shape[1]):
        for j in range(0, features.shape[0]):
            for n in range(1,dWin.size):
                
                #Check: if index range is negative, then set to first index
                if ((j-n) < 0):
                    lowerBound = 0
                else:
                    lowerBound = j-n
                
                #Check: if index range > sigLen, then set to last index
                if ((j+n) > features.shape[0]-1):
                    upperBound = features.shape[0] - 1
                else:
                    upperBound = j+n
                    
                #Sum values to return deltas
                deltaArray[j,i] = deltaArray[j,i] + (n * (features[upperBound,i] - features[lowerBound,i]))
            
            #normalize and store
            deltaArray[j,i] = deltaArray[j,i]/normDelta       
    return deltaArray #return

def getDiagonals(covMat, numDiags):
#Compute covariance diagonals
# ============================================================================    
    
    diags = []
    #Consider only diagonals of interest from covariance matrix and create a vector
    for i in range(0, numDiags):
        currDiag = np.diag(covMat, i)
        diags.extend(currDiag);
    
    return np.asarray(diags) #return as array

#Test  
# =============================================================================
# filepath = "D:/MIR/Assignment3/audio3/piano_test.wav"
# win_size = 1024
# hop_size = 512
# min_freq = 86
# max_freq = 8000
# num_mel_filts = 40
# n_dct = 15
# num_diags = 6
# length_deltaWin = 9
# features = output_features(filepath, win_size, hop_size, min_freq, max_freq, num_mel_filts, n_dct, num_diags, length_deltaWin)
# 
# =============================================================================

