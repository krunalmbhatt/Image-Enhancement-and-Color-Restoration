''''
In this code we are implementing the combined normal 3-channel histogram and adaptive histogram approach.
''''

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
    
    #masked = frame - img_inv
    #cv2.imshow("Masked",masked)

    '''
    The image is inverted and now will be split in 3 channels b g and r respectively.
    '''

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


    '''
    Here now we use the image which has been equalized seprately in 3 b g r channels and then apply adaptive histogram.
    Change the cliplimit =5.0 to other value to observe change in output.
    Different channels commented can be uncommented to observe its effect in other h,s,or v channels.
    '''
    
    #CLAHE
    hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    #create 3 color channels
    h,s,v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]

    #APPLY CLAHE
    clahe = cv2.createCLAHE(clipLimit = 5.0, tileGridSize = (8,8))
    v = clahe.apply(v)
    #h = clahe.apply(h)
    #s = clahe.apply(s)

    hsv_img = np.dstack((h,s,v))

    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    
    #img1=image_eq

    cv2.imwrite("D:/1_MY_WORK/Andromeida/code/UIEB DATASET/output/output_eqclahe/"+name, rgb)
    

    #cv2.imshow("ENHANCED", img1)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
