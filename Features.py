import scipy
import numpy as np
from skimage.measure import compare_ssim as ssim

def features(img1,img2):
    f1 = Hausdroff(img1, img2)
    f2 = tanimoto(img1,img2)
    f3 = mse(img1,img2)
    f4 = ssim(img1,img2)
    f5 = pearson(img1,img2)
    f6 = kendall(img1,img2)
    return np.array([f1,f2,f3,f4,f5[0],f5[1],f6[0],f6[1]])

def mse(img1,img2):
    err = np.linalg.norm((img1.astype("float") - img2.astype("float")) ** 2)
    return err

def kendall(img1,img2):
    err = scipy.stats.kendalltau(img1, img2, initial_lexsort=True)
    return err

def pearson(img1,img2):
    return scipy.stats.pearsonr(img1, img2)

def Hausdroff(img1,img2):
    A = img1.reshape(-1,1)
    B = img2.reshape(-1,1)
    D = scipy.spatial.distance.cdist(A,B, 'euclidean')
    # Hausdorff distances is not symmetric
    d1 = np.max(np.min(D, axis=1))
    d2 = np.max(np.min(D, axis=0))
    return (d1+d2)/2.0

def tanimoto(a,b):
    c=[v for v in a if v in b]
    return float((len(c))/(len(a)+len(b)-len(c)))



