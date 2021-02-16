from sys import path
from numpy.core.numeric import cross
import pydicom as pyd
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
import matplotlib.pyplot as plt
from scipy import signal 
from scipy.fft import fft2, fftshift,ifft2,ifftshift

def centert(x,y): 
    #used for centering 
    return -1**(x+y)
def norm_step(img): 
    xax = np.arange(0,img.shape[0])
    yax = np.arange(0,img.shape[1]) 
    x,y = np.meshgrid(xax,yax)
    img = img * centert(x,y).T 
    return img 

def calc_fft(x:np.ndarray,center=False): 
    ff = fft2(x)
    if center:
        ff =  fftshift(x)
    return ff
def freq2spatial(img): 
    #appy  inverse frequency domain  and center the image 
    img =  norm_step(np.real(ifft2(img)))
    return img


def gen_filt(im1,rad=1): 
    x,y = im1.shape[0],im1.shape[1]
    filt = np.ones((x,y)) 
    #find the center of the image 
    c_x,c_y =  filt.shape[0]/2,filt.shape[1]/2
    # Use meshgrid approach to get distances from center
    xax = np.arange(0,x)
    yax = np.arange(0,y)
    x,y = np.meshgrid(xax,yax)
    #this calculates distances from center of image
    huv  = lambda x,y,cx,cy,rad : np.sqrt((x-cx)**2 + (y-cy)**2)
    huv = huv(x,y,c_x,c_y,rad)
    #create butteworth filter
    outfilt = 1/ (1+ (huv/rad)**(2*5)).T 
    return  outfilt

def my_fft(img):
    #helper method for applying fft 
    return calc_fft(norm_step(img),center=False) 
def freq_domain(img:np.ndarray,filt='low',rad=5,plot=False):
    #do padding with the image processing 
    x,y = img.shape 
    p,q = (2*x,y*2)
    img =  np.pad(img,((0,p-x),(0,q-y)) ) 
    img_f = my_fft(img)
    filt_f = gen_filt(img_f,rad=rad) 
    if filt=='low':
        output = img_f * filt_f  
    else: 
        output = img_f * (1- filt_f ) 
    if plot: 
        plt.figure()
        plt.subplot(121)
        if filt =='low':
            plt.plot(img_f[112,112:224])
            plt.plot(output[112,112:224])
            plt.legend(['original','filtered'])
            plt.subplot(122)
            plt.plot((filt_f)[112,112:224])
            plt.legend(['Gain'])
        else: 
            plt.plot(img_f[112,112:224])
            plt.plot(output[112,112:224])
            plt.legend(['original','filtered'])
            plt.subplot(122)
            plt.plot((1-filt_f)[112,112:224])
            plt.legend(['Gain'])
    return  output
def freq_images(img1,img2,rad_1=13,rad_2=13,plot=False): 
    x,y = img1.shape
    img1_f = freq_domain(img1,filt='low',rad=rad_1,plot=plot) # around 4-8 is when this dude dissapears 
    img2_f = freq_domain(img2,filt='high',rad=rad_2,plot=plot)
    plt.show()
    output = freq2spatial(img1_f + img2_f)[0:x,0:y]
    return output
def viz_freqs(plot): 
    im1 = np.array(Image.open('Image1.png').resize((112,112)  )) #albert 
    im1 = im1/np.max(im1[:])
    im2 = np.array(Image.open('Image2.png').resize((112,112) ) ) #mary
    im2 = im2/np.max(im2[:])
    merged = np.zeros((112,112,3))
    for i in range(0,3):
        merged[:,:,i] = freq_images(im1[:,:,i],im2[:,:,i],rad_1=115,rad_2=115,plot=plot) 
    plt.imshow(merged)
    plt.show() 
def viz_freqs2(plot): 
    im1 = np.array(Image.open('lion.jpg').resize((112,112)  )) 
    im1 = im1/np.max(im1[:])
    im2 = np.array(Image.open('jag.png').resize((112,112) ) )
    im2 = im2/np.max(im2[:])
    merged = np.zeros((112,112,3))
    for i in range(0,3):
        merged[:,:,i] = freq_images(im1[:,:,i],im2[:,:,i],rad_1=90,rad_2=175,plot=plot) 
    plt.imshow(merged)
    plt.show() 
if __name__=='__main__':
    #change to true so it shows plots of gains and the outputs. 
    #change to false input to hide gain plots 
    # does the einstien image 
    viz_freqs(True)
    # does the lion and jaguar images 
    #viz_freqs2(True)