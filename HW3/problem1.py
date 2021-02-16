from sys import path
from numpy.core.numeric import cross
import pydicom as pyd
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 


def pad_img(img,kernel,pad='zero'): 
    x,y = img.shape 
    #figure out by how much to extend image by 
    k_x,k_y = np.ceil(kernel.shape[0]/2).astype(int),np.ceil(kernel.shape[1]/2).astype(int)
    #copy over the values
    copy = np.zeros((x+k_x,y+k_y))
    k_x,k_y = np.floor(kernel.shape[0]/2).astype(int),np.floor(kernel.shape[1]/2).astype(int)
    for i in range(0,x):
        for j in range(0,y):
            copy[i+k_x,j+k_y] = img[i,j]
    if pad =='same': 
        copy[0:k_x,:] = copy[k_x,:]
        copy[-1*k_x:,:] = copy[-1*k_x,:]
        for e in range(0,k_y): 
            copy[:,e]= copy[:,k_y]
        for e in range(-1*k_y,-1):
            copy[:,e] = copy[:, e]
    return copy

def convolution(mat,k,pad='zero'): 
    kernel = k.T # required kernel flip 
    divisor =  np.sum(kernel[:]) 
    if divisor ==0:
        divisor = 1 
    out= cross_corr(mat,kernel)/divisor
    return out


def cross_corr(mat,kernel,norm_input=False): 
    #traditional padding 
    p_img = pad_img(mat,kernel=kernel)
    #figure out dimensions  
    dx,dy = kernel.shape 
    k_x,k_y = np.floor(kernel.shape[0]/2).astype(int),np.floor(kernel.shape[1]/2).astype(int)
    out_img = np.zeros(p_img.shape)
    #iterate through the image
    for i in range(k_x,p_img.shape[0]-k_x): 
        for j in range(k_y,p_img.shape[1]-k_y):
            flat_t = kernel.flatten() 
            flat_p_img = p_img[i-k_x:i-k_x+dx ,j-k_y:j-k_y+dy].flatten()
            #normalize input to get -1 to 1 output 
            if norm_input: 
                flat_p_img = flat_p_img-np.mean(flat_p_img)
                flat_p_img = (flat_p_img )/np.linalg.norm(flat_p_img)
            #zero norm occurs when vector is zero. output is zero 
            if np.any( np.isnan(flat_p_img)):
                out_img[i,j]= 0 
            else:
                out_img[i,j]= np.dot(flat_t , flat_p_img) 
    #crop out the true output
    final_out = out_img[k_x:k_x+mat.shape[0],k_y:k_y+mat.shape[1] ]
    # plus one is due to range being [0,x) not inclusive
    return final_out

def my_imfilt(img,kernel,pad='same'): 
    if len(img.shape) >2 and  img.shape[2] >1: 
        output = np.zeros(img.shape)
        for e in range(0,img.shape[2]):
            output[:,:,e] = convolution(img[:,:,e],kernel,pad=pad)
    else: 
        output = convolution(img,kernel,pad=pad)
    return output

def problem1(): 
    img =   np.array(Image.open('jag.png'))
    kernel = np.ones((9,7))
    output = my_imfilt(img,kernel=kernel)
    plt.figure()
    plt.subplot(131)
    plt.imshow(output[:,:,0])
    plt.subplot(132)
    plt.imshow(output[:,:,1])
    plt.subplot(133)
    plt.imshow(output[:,:,2])
    plt.show()

if __name__=="__main__":
    problem1()
