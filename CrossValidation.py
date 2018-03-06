from sklearn.ensemble import RandomForestClassifier
import numpy as np
import Call_Baseline
# =============================================================================

#For 5s
SAMPLES = 2000
FILESPERCLASS = 400

#Feature selection
#feature = 'mfcc'
feature = 'audiot'
#feature = 'combo'

#Feature space size selection
if feature == 'audiot':
    FEATURES = 255
elif feature == 'mfcc':
    FEATURES = 279
#if combo, uncomment below line
#FEATURES = 534    
    
X, y = Call_Baseline.organizeESC50(feature)

#Test code
#np.save('X_at128',X)
#Xm = np.load('X_mfcc512.npy')
#X = np.load('X_at128.npy')
#X = np.concatenate((Xm,Xat),axis = 1)
#y = np.load('y_v3.npy')

#Manual test and train splits for accurate feature comparison
y = y.reshape(len(y),1)
y_indices = np.arange(SAMPLES)
y_indices = y_indices.reshape(len(y_indices),1)
y = np.concatenate((y,y_indices), axis = 1)
 
np.random.shuffle(y)
y = y.astype(int)
 
y1,y2,y3,y4,y5 = np.vsplit(y,5)

X1 = np.zeros((FILESPERCLASS, X.shape[1]))
X2 = np.zeros((FILESPERCLASS, X.shape[1]))
X3 = np.zeros((FILESPERCLASS, X.shape[1]))
X4 = np.zeros((FILESPERCLASS, X.shape[1]))
X5 = np.zeros((FILESPERCLASS, X.shape[1]))

for i in range(0, len(y1)):
    X1[i,:] = X[y1[i,1],:]
    X2[i,:] = X[y2[i,1],:]
    X3[i,:] = X[y3[i,1],:]
    X4[i,:] = X[y4[i,1],:]
    X5[i,:] = X[y5[i,1],:]
 
#Using unique splits for test and train
#Using X1 as test
X_test = X1
X_train = np.array(np.concatenate((X2, X3, X4, X5),axis=0))
y_test = np.delete(y1,1,1)
y_train = np.array(np.concatenate((y2, y3, y4, y5),axis=0))
y_train = np.delete(y_train,1,1)

#Using X2 as test
#X_test = X2
#X_train = np.array(np.concatenate((X1, X3, X4, X5),axis=0))
##y_test = np.delete(y2,1,1)
#y_train = np.array(np.concatenate((y1, y3, y4, y5),axis=0))
#y_train = np.delete(y_train,1,1)
    
#Using X3 as test
#X_test = X3
#X_train = np.array(np.concatenate((X1, X2, X4, X5),axis=0))
#y_test = np.delete(y3,1,1)
#y_train = np.array(np.concatenate((y1, y2, y4, y5),axis=0))
#y_train = np.delete(y_train,1,1)

#Using X4 as test
#X_test = X4
#X_train = np.array(np.concatenate((X1, X2, X3, X5),axis=0))
#y_test = np.delete(y4,1,1)
#y_train = np.array(np.concatenate((y1, y2, y3, y5),axis=0))
#y_train = np.delete(y_train,1,1)

#Using X5 as test
#X_test = X5
#X_train = np.array(np.concatenate((X1, X2, X3, X4),axis=0))
#y_test = np.delete(y5,1,1)
#y_train = np.array(np.concatenate((y1, y2, y3, y4),axis=0))
#y_train = np.delete(y_train,1,1)

#Build classifier
clf = RandomForestClassifier(max_features = 16, max_depth=9, random_state=0)  
clf.fit(X_train, y_train)
 
y_pred = np.zeros(np.shape(y_test))

#Classification Accuracy Calculation
numPred = 0; numCorrPred = 0
c1 = 0;c2 = 0;c3 = 0;c4 = 0;c5 = 0
c1ac = 0;c2ac = 0;c3ac = 0;c4ac = 0;c5ac = 0

for i in range(0,len(y_pred)):
    
    y_pred[i] = clf.predict([X_test[i,:]])
    
    #Calculate class-wise accuracy
    if y_test[i] == 1:
        c1 = c1 + 1    
        if y_pred[i] == y_test[i]:
            c1ac = c1ac + 1
    if y_test[i] == 2:
        c2 = c2 + 1    
        if y_pred[i] == y_test[i]:
            c2ac = c2ac + 1
    if y_test[i] == 3:
        c3 = c3 + 1    
        if y_pred[i] == y_test[i]:
            c3ac = c3ac + 1
    if y_test[i] == 4:
        c4 = c4 + 1    
        if y_pred[i] == y_test[i]:
            c4ac = c4ac + 1
    if y_test[i] == 5:
        c5 = c5 + 1    
        if y_pred[i] == y_test[i]:
            c5ac = c5ac + 1
            
#Calculate overall accuracy
numCorrPred = c1ac+c2ac+c3ac+c4ac+c5ac
numPred = c1+c2+c3+c4+c5
if numPred == 0:
    print('ERROR!')
else:
    acc = numCorrPred/numPred*100

if c1 == 0:
    print('Class1 (Animals) Accuracy: 0 %')
else:
    print('Class1 (Animals) Accuracy:', round(c1ac/c1*100,2),'%')
if c2 == 0:
    print('Class2 (Natural) Accuracy: 0 %')
else: 
    print('Class2 (Natural) Accuracy:', round(c2ac/c2*100,2),'%')
if c3 == 0:
    print('Class3 (Human) Accuracy: 0 %')
else: 
    print('Class3 (Human) Accuracy:', round(c3ac/c3*100,2),'%')
if c4 == 0:
    print('Class4 (Interior) Accuracy: 0 %')
else: 
    print('Class4 (Interior) Accuracy:', round(c4ac/c4*100,2),'%')
if c5 == 0:
    print('Class5 (Exterior) Accuracy: 0 %')
else: 
    print('Class5 (Exterior) Accuracy:', round(c5ac/c5*100,2),'%')

print('          Overall Accuracy:', round(acc,2), '%')
