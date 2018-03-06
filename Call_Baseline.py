import Baseline_MFCC_FeatExt
import textureFeatures
import numpy as np
import glob
import os 
# =============================================================================

SAMPLES = 2000

def organizeESC50(feat):
# =============================================================================
    
    #Defining critical paramters    
    win_size = 512
    hop_size = 256
    min_freq = 86
    max_freq = 8000
    num_mel_filts = 40
    n_dct = 15
    num_diags = 6
    len_delWin = 9
    num_subbands = 18
    num_octaveBands = 6
    
    #Feature-type check
    if feat == 'audiot':
        FEATURES = 255
    elif feat == 'mfcc':
        FEATURES = 279
    #if combo, uncomment below line
    #FEATURES = 534  
    
    Y = np.zeros(SAMPLES)
    X = np.zeros([SAMPLES, FEATURES])
    
    os.chdir('..')
    os.chdir('Sorted-ESC-50-5')
    
    inc = 0
    
    classNum = 1
    
    #Traversing directories and extracting features
    for subclass in os.listdir():
        if os.path.isdir(subclass):
            os.chdir(subclass)
            for filename in glob.glob('*.wav'):
                if feat == 'audiot':
                    currFeatVec = textureFeatures.output_features(filename, win_size, hop_size, min_freq, max_freq, num_subbands, num_diags, num_octaveBands)
                elif feat == 'mfcc':
                    currFeatVec = Baseline_MFCC_FeatExt.output_features(filename, win_size, hop_size, min_freq, max_freq, num_mel_filts, n_dct, num_diags, len_delWin)
                X[inc,:] = currFeatVec
                Y[inc] = classNum
                inc = inc+1
            print('NEW DIR!')
            os.chdir('..')
            classNum = classNum + 1
            
    #Error handling
    X = np.nan_to_num(X)
    return X,Y