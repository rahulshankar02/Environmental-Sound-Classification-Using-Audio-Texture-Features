import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import Conversion as convert
# =============================================================================

def output_features(filepath, win_size, hop_size, min_freq, max_freq, num_subbands, num_diags, num_octaveBands):
# ============================================================================     
    fs, x_t = scipy.io.wavfile.read(filepath)
    x_t = x_t/np.max(abs(x_t))
    
    #Handle zero padding extensions
    delIdx = []
    for i in range (0, len(x_t)):
        if (x_t[i] == 0):
            delIdx.append(i) 
    
    x_t = np.delete(x_t, delIdx)
            
    logmag_spectrum, fs_fft = auditoryFilterbank(x_t, fs, win_size, hop_size, min_freq, max_freq, num_subbands)
    moments, var = getMoments(logmag_spectrum)
    moments = moments.flatten('C')
    covMat = np.corrcoef(logmag_spectrum, rowvar = False) #get covariance matrix
    covDiags = getDiagonals(covMat, num_diags) #grab diagonals
    modSpec = modulationSpectrum(logmag_spectrum, num_octaveBands, fs_fft, var).flatten('C')
    
    return np.concatenate((moments, covDiags, modSpec), axis = 0); #returns

def modulationSpectrum(magSpec, octaves, fs, var):
# ============================================================================
    mod_spectrum = np.zeros((magSpec.shape[1] , octaves)) 
    
    for i in range(0, magSpec.shape[1]):
        out = abs(np.fft.rfft(magSpec[:,i]))
        freq = np.fft.rfftfreq(magSpec.shape[0], d = fs)
        #s, f, t, image = plt.specgram(logmag_spectrum[:,i], Fs = fs, window = windowF, nfft = win_size, noverlap = overlap, mode = 'magnitude')
        for j in range(0, len(freq)):
            if (freq[j] >= .5 and freq[j] <= 1):
                mod_spectrum[i, 0] = mod_spectrum[i,0] + out[j]
            elif (freq[j] > 1 and freq[j] <= 2):
                mod_spectrum[i, 1] = mod_spectrum[i,1] + out[j]    
            elif (freq[j] > 2 and freq[j] <= 4):
                mod_spectrum[i, 2] = mod_spectrum[i,2] + out[j] 
            elif (freq[j] > 4 and freq[j] <= 8):
                mod_spectrum[i, 3] = mod_spectrum[i,3] + out[j] 
            elif (freq[j] > 8 and freq[j] <= 16):
                mod_spectrum[i, 4] = mod_spectrum[i,4] + out[j]
            elif (freq[j] > 16 and freq[j] <= 32):
                mod_spectrum[i, 5] = mod_spectrum[i,5] + out[j]
        mod_spectrum[i,:] = mod_spectrum[i,:] / var[i]
        
    return mod_spectrum

def auditoryFilterbank(x_t, fs, win_size, hop_size, min_freq, max_freq, num_subbands):
# ============================================================================ 
        
    overlap = win_size-hop_size
    windowF = np.hamming(win_size)
    s, f, t, image = plt.specgram(x_t, Fs = fs, window = windowF, NFFT = win_size, noverlap = overlap, mode = 'magnitude')
    #s = abs(s);
    fs_fft = hop_size/fs
    min_mel = convert.hz2mel(min_freq)
    max_mel = convert.hz2mel(max_freq)
    mel_c = np.linspace(min_mel,max_mel,num=num_subbands+2)
    freq_c = convert.mel2hz(mel_c)
    
    #Rounding to closest bin
    bin_c = convert.find_nearest(f, freq_c)
    bank = np.zeros((np.size(f),num_subbands))
    bank_norm = np.zeros((np.size(f),num_subbands))

    #Constructing individual filterbank
    for i in range(0,num_subbands):
        pre = np.zeros(((bin_c[i])), dtype=np.int)
        up = np.linspace(0,1,num=(bin_c[i+1]-bin_c[i])+1)
        down = np.linspace(1,0,num=(bin_c[i+2]-bin_c[i+1])+1)
        down = down[1:]
        post = np.zeros(((np.size(f)-bin_c[i+2])-1), dtype=np.int)
        bank[:,i] = np.concatenate([pre,up,down,post])
        bank_norm[:,i] = np.divide(bank[:,i],(np.sum(bank[:,i])))
        #plt(bank_norm)
    
    #Log magnitude & multiplication with filterbank
    logmag = 20*np.log10(np.dot(np.transpose(s),(bank_norm)))
    
    return logmag, fs_fft
 
def getMoments(magSpec):
# ============================================================================         
    peakVal_perSubband = np.amax(magSpec, axis = 0) #peak value per subband
    
    av = np.zeros(magSpec.shape[1])
    var = np.zeros(magSpec.shape[1])
    skew = np.zeros(magSpec.shape[1])
    kurtosis = np.zeros(magSpec.shape[1])
    sigma_squared = np.zeros(magSpec.shape[1])
    
    for i in range(0, magSpec.shape[1]):
        cleanSubband = cleanSubbands(magSpec[:,i],peakVal_perSubband[i])
        av[i] = np.average(cleanSubband)
        sigma_squared[i] = scipy.stats.tvar(cleanSubband)
        var[i] = sigma_squared[i] / np.power(av[i],2)
        skew[i] = scipy.stats.skew(cleanSubband)
        kurtosis[i] = scipy.stats.kurtosis(cleanSubband)
    
    return np.concatenate((av,var,skew,kurtosis), axis = 0), sigma_squared
        
 
def cleanSubbands(subband, peakVal):
# ============================================================================     
    cleanSubband = []
    for i in range(0, len(subband) ):
        if subband[i] >= (peakVal - 40):
            cleanSubband.append(subband[i])
    return np.asarray(cleanSubband)
            
#grab covariance diagonals
def getDiagonals(covarianceMat, numDiags):
# ============================================================================     
    #grab only diagonals of interest from covariance matrix and create a vector
    diags = []
    
    for i in range(1, numDiags): #exclude leading diagonal
        currDiag = np.diag(covarianceMat, i)
        diags.extend(currDiag);
    
    return np.asarray(diags) #return as array

#Test
# =============================================================================
#filepath = "/Users/Rahul/Documents/School/MIR/MIR Project/Sorted-ESC-50/01_Animal/1-100032-A2.wav"
#win_size = 1024
#hop_size = 512
#min_freq = 86
#max_freq = 8000
#num_subbands = 18
#num_diags = 6
#num_octaveBands = 6
 
#features = output_features(filepath, win_size, hop_size, min_freq, max_freq, num_subbands, num_diags, num_octaveBands)
# 
# =============================================================================
