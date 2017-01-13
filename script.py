import numpy as np
import glob
from sklearn.decomposition import PCA
from PIL import Image
from numpy import *
import os
import matplotlib.pyplot as plt

def loadpngs():
    trainData = []
    trainLabels = []
    
    for filename in glob.glob('/home/gpietro/PhD/examsUB/610-DL/Handwritten-and-data/Cursive/imagesMod/*.png'):
        str=os.path.basename(filename)
        tag=str.split('_')[0]
        tag=int(''.join(i for i in tag if i.isdigit()))
        img=Image.open(filename).convert('L')#black and white
        arr = np.array(img)
        trainData.append(arr)
        trainLabels.append(tag)
        img.close()
    return((np.asarray(trainData), np.asarray(trainLabels)))

def pcaRed(img):
    mu = np.mean(img, axis=0)
    pca = PCA()
    pca.fit(img)
    return(pca)

def createPairs(k):
    x,y=loadpngs()
    perm1=np.random.permutation(range(k))
    p1=x[perm1]
    t1=y[perm1]
    perm2=np.random.permutation(range(k))
    p2=x[perm2]
    t2=y[perm2]
    return (((p1,t1),(p2,t2)))
    
    
def diff(img1, tag1, img2, tag2):
    z=img1-img2
    if (tag1==tag2):
        tag=1
    else:
        tag=0
    return(z,tag)

def learn(zs):
    pca=[]
    for diff in zs:
        tmp=(PCA().fit(diff.reshape(40,40)).components_)
        tmp=tmp[:,:25]
        tmp=tmp.flatten()
        pca.append(tmp)
    return(pca)

def start(k):
    (x1,t1),(x2,t2) = createPairs(k)
    z=[]
    t=[]
    for i in range(len(x1)):
        h,g = diff(x1[i],t1[i],x2[i],t2[i])
        z.append(h)
        t.append(g)       
    pcas=np.asarray(learn(np.asarray(z)))
    np.save('X_Data', pcas)
    np.save('Y_Data', t)
    return(pcas,t)

start(150)
