import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt 

def Gradient_Magnitude(img): 
    #Calculate gradient magnitudes using  gx, gy filters
    # each filter calculates corresponding directions intensity
    gx =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    gy = gx.T 
    #calculate magnitude using sqrt( x**2 + y**2)
    output =  (convolve(img,gx)**2  + convolve(img,gy)**2 )**.5
    return output

def Laplacian_(img): 
    #apply laplacian filter onto our image
    filt = np.array([[0,1,0],[1,-4,0],[0,1,0] ])
    output = convolve(img,filt)
    return output

def gen_filter_bank(): 
    #filter for the following directions 
    main_axis = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) #up down 
    main_axis_y = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]) #left right 
    lr_down_diag = np.array([[1,0,0],[0,0,0],[0,0,-1]]) # one diagnonal. 
    lr_up_diag = np.array([[0,0,-1],[0,0,0],[0,0,1]])#other diagonal 
    return [main_axis, main_axis_y, lr_down_diag,lr_up_diag]

def detect_crossing(arr,filt,threshold =0.01 ): 
    output = (convolve(arr,filt) >= threshold).astype(int) 
    return output 
def Zero_Crossings(arr): 
    #apply multiple filter banks that take difference along a region 
    output = np.zeros(arr.shape) 
    for e in gen_filter_bank(): 
        output +=  detect_crossing(arr,e,threshold=0.4) #threshold is meant to improve quality of output. This accounts for 
        # the fact some edge changes may not be exact. 
    output = (output >= 1).astype(int) 
    return output

def Laplacian_Edge_Detection(img,mag_t=.5): 
    #apply edge detection by calculating magnitudes then logical and with laplacian zero crossings 
    magni = Gradient_Magnitude( img)
    magni = ( (img) > mag_t) .astype(int)
    lap = Laplacian_(img)
    cross = Zero_Crossings(lap) 
    return np.logical_and( magni,cross)

def main(): 
    img = np.array(Image.open('./main_img.png').convert(mode="L")) 
    img = img/np.max(img[:])
    grad_mag = Gradient_Magnitude(img) 
    plt.figure(1)
    plt.subplot(151) 
    plt.imshow(img) 
    plt.title('original') 
    plt.subplot(152)  
    plt.imshow(grad_mag) 
    plt.title('Gradient') 
    plt.subplot(153)
    lap = Laplacian_(img)
    plt.imshow(lap) 
    plt.title('Laplacian') 
    crossings = Zero_Crossings(lap) 
    plt.subplot(154)
    plt.imshow(crossings)
    plt.title('Crossings') 
    plt.subplot(155)
    plt.imshow(Laplacian_Edge_Detection(img,mag_t=.5))
    plt.title('Laplacian Edge Detection') 
    plt.show()

if __name__=="__main__":
    main()