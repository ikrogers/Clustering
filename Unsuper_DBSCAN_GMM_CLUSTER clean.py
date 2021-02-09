import cv2
import numpy as np
import glob
from sklearn.cluster import DBSCAN
from sklearn import metrics
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as img 
import matplotlib.colors as clr
from sklearn.mixture import GaussianMixture
import pandas as pd
import random as random
# #############################################################################

#Collect photos
imgArray = []
for f in glob.iglob("C:/Users/Ilya Rogers/Documents/Research/Sample/*"):
    imgArray.append(f)
imgArray = np.array(imgArray)

for i in range(imgArray.size):
    #Process images based on pixel color
    im = Image.open(imgArray[i])
    pix = im.load()
    pixelLoc = []
    pixelColorType = []

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            coords = tuple([x,y])
            pxRGB = im.getpixel(coords)
            if pxRGB[0] > 150 and pxRGB[1] <= 50 and pxRGB[2] >= 0 : #red Spectrum RGB 1 controls how much red I see - Higher the more
                pixelLoc.append(coords)
            elif pxRGB[0] >= 0 and pxRGB[1] > 150 and pxRGB[2] <=80: #Green spectrum
                pixelLoc.append(coords)
            elif pxRGB[0] <= 80 and pxRGB[1] >= 0 and pxRGB[2] > 100: #Blue Spectrum
                pixelLoc.append(coords) 
                
    try:
        X = pixelLoc
        #print(pixelColorType)
        ss = StandardScaler()
        ss.fit(X)
        X_orig = ss.transform(X)
        X = X_orig
        passed_points_indeces = range(len(X_orig))

        X_orig = X_orig[passed_points_indeces]
               
        X_passed = ss.inverse_transform(X_orig)
    except: 
        print ("Fitting failed")
        quit()

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps= .2, min_samples=70).fit(X) #min samples = data pts eps = distance bw points to = cluster
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    #print(str(i)+ ' ' +str(n_clusters_))
    print('Sample: '+str(i)+ ' Est # of clusters: ' +str(n_clusters_)+' Est Noise: ' + str(n_noise_))

    #If 0 clusters assume there is only one present
    if n_clusters_ == 0:
        n_clusters_ = 1

    # #############################################################################
    # Compute GMM
    X_passed = ss.inverse_transform(X_orig) #this restores original XY Coords
    gmm = GaussianMixture(n_components=n_clusters_).fit(X_passed)

    #Predictions from gmm
    labels = gmm.predict(X_passed)
    frame = pd.DataFrame(X_passed)
    frame['cluster'] = labels
    frame.columns = ['Weight', 'Height', 'cluster']

    #Dynamic Color based off # of clusters
    colors = []
    rgbl=[255,0,0]
    
    for j in range(n_clusters_):
        colors.append(random.shuffle(rgbl))  

    plt.subplot(4, 5, i+1)
    
    #Build the sublopt
    for k in range(n_clusters_):#iterate in range of clusters
        data = frame[frame["cluster"]==k]
        plt.scatter(data["Weight"],data["Height"],c=colors[k])
        
    plt.title('S = ' +str(i+1) + ' Est clusters: %d' % n_clusters_)

    plt.axis('off') 

plt.show()
print("Done")
