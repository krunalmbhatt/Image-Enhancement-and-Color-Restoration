'''
THIS IS THE ORIGINAL HISTOGRAM EQUALIZATION CODE. HERE THE IMAGE IS TAKEN AND INVERTED.
THE COMMENTED PORTIONS ARE NOT REMOVED IN-CASE IF NEEDED TO CHANGE THE PIPELINE.
'''

import cv2
import numpy as np

bins = 256

#take video as input form camera
#cap = cv2.VideoCapture(0)

#cap = cv2.imread('2.jpg')

print('PROCESSING LIVE FEED')

c=1

while(True):
    #path = cv2.imread('D:\\1_MY_WORK\\Andromeida\\code\\UIEB DATASET\\raw-890')
    name = "1 ("+str(c)+").png"
    cap = cv2.imread(name)
    c=c+1
    #_,img = cap.read() 
    #frame = img

    frame = cap
    
    #frame=cv2.resize(frame,(500,500)) #resize the frame
    #cv2.imshow("ORIGINAL", frame) #shows unenhanced image

    img_inv = cv2.bitwise_not(frame)
    #cv2.imshow("INVERTED", img_inv)
    
    #Get the dark channel in inverted image
    
    
    #masked = frame - img_inv
    #cv2.imshow("Masked",masked)

    #individual 3 channel histogram equalization
    b,g,r=cv2.split(img_inv)

    
    ###BLUE CHANNEL
    b_flattened = b.flatten()
    b_hist = np.zeros(bins)
    for pix in b:
            b_hist[pix] += 1
    cum_sum = np.cumsum(b_hist)
    norm = (cum_sum - cum_sum.min()) * 180
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    b_eq = uniform_norm[b_flattened]
    # reshaping the flattened matrix to its original shape
    b_eq = np.reshape(a=b_eq, newshape=b.shape)
    b_eq=np.uint8(b_eq)


    ###GREEN CHANNEL
    g_flattened = g.flatten()
    g_hist = np.zeros(bins)
    for pix in g:
            g_hist[pix] += 1

    cum_sum = np.cumsum(g_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    g_eq = uniform_norm[g_flattened]
    # reshaping the flattened matrix to its original shape
    g_eq = np.reshape(a=g_eq, newshape=g.shape)
    g_eq=np.uint8(g_eq)


    ###RED CHANNEL
    r_flattened = r.flatten()
    r_hist = np.zeros(bins)
    for pix in r:
            r_hist[pix] += 1

    cum_sum = np.cumsum(r_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    r_eq = uniform_norm[r_flattened]
    # reshaping the flattened matrix to its original shape
    r_eq = np.reshape(a=r_eq, newshape=r.shape)
    r_eq=np.uint8(r_eq)

    image_eq=cv2.merge((b_eq,g_eq,r_eq))
    img1= cv2.bitwise_not(image_eq)
    #img1=image_eq

    cv2.imwrite("D:/1_MY_WORK/Andromeida/code/UIEB DATASET/output/"+name, img1)
    

    #cv2.imshow("ENHANCED", img1)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
