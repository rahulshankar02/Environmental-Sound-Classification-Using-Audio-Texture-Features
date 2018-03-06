import numpy as np

def hz2mel(hzval):
#Convert a vector of values in Hz to Mels.
# ============================================================================    
 
    melval = 1127.01028*np.log(np.divide(hzval,700) + 1)
    return melval



def mel2hz(melval):
#Convert a vector of values in Hz to Mels.
# ============================================================================    

    hzval = 700*(np.exp(np.divide(melval,1127.01028))-1)
    return hzval


def find_nearest(reference, target):
#Find indices of nearest values in a reference array to a target array.
# ============================================================================     
    
    L_target = len(target)
    L_reference = len(reference)
    diff_matrix = np.transpose(np.tile(reference,(L_target,1))) - np.tile(target,(L_reference,1))
    abs_matrix = np.array(np.absolute(diff_matrix))
    
    return np.argmin(abs_matrix,axis=0)