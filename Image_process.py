import numpy as np
import glob
from PIL import Image
import os
import PIL.ImageOps
from PIL import ImageFilter

# Size of the image to which an image is compressed
size = 40,40
path_save = '/Users/Hemanth/Documents/My UB/Deep learning /Project 1/Modified Images/'

def Loadpngs():
    trainData = []
    trainLabels = []

    for filename in glob.glob('/Users/Hemanth/Documents/My UB/Deep learning /Project 1/Test/*.png'):
        str = os.path.basename(filename)
        tag = str.split('_')[0]
        tag = int(''.join(i for i in tag if i.isdigit()))
        img = Image.open(filename).convert('L')  # black and white
        inverted_image = PIL.ImageOps.invert(img)
        B = inverted_image.resize(size, Image.ANTIALIAS) # resize the image
        # B = B.filter(ImageFilter.EDGE_ENHANCE_MORE)
        file_path = os.path.join(path_save, 'N' + os.path.basename(filename))
        B.save(file_path)
        C = np.array(B)# convert to 1's and 0's (bit image)
        C = C.flatten()
        C = (C-min(C))/(max(C)-min(C))
        arr = np.array(C)
        trainData.append(arr)
        trainLabels.append(tag)
        img.close()
    return ((np.array(trainData), np.array(trainLabels)))

[X,Y] = Loadpngs()

np.save('X_Data',X)
np.save('Y_Data',Y)

print(X.shape)
