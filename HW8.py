from PIL import Image 
import numpy as np 
import  matplotlib.pyplot as plt


def update_counts(counts,img,i,j,direction=0):
    val = img[i,j]
    if direction ==360:
        direction =0
    if direction==0 and j+1 < img.shape[0]:
        counts[val,img[i,j+1]] +=1 
    if direction ==90 and i-1 >=0:
        counts[val,img[i-1,j]] +=1
    if direction ==180 and j-1 >=0:
        counts[val,img[i,j-1]] +=1
    if direction ==270 and i+1 < img.shape[1]:
        counts[val,img[i+1,j]] +=1
def glcm(img,num_levels,direction): 
    counts = np.zeros((num_levels,num_levels))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            update_counts(counts,img,i,j,direction=direction) 
    return counts


def gen_lbp_val(region):
    middle = region[1,1]
    region = region.flatten()
    access_order = [0,1,2,5,8,7,6,3] # these are the indeces we care about and we read them in clockwise motion 
    binary = ""
    for idx in access_order:
        if middle < region[idx] :
            binary= binary +"0"
        else:
            binary = binary + "1"
    conv = int(binary,2)
    return conv

def  extract_local_binary_pattern(img):
    lbp = np.zeros((img.shape[0]-1,img.shape[1]-1))
    for i in range(1,img.shape[0]-1): 
        for j in range(1,img.shape[1]-1):
            lbp[i,j] = gen_lbp_val(img[i-1:i+2,j-1:j+2])
    return lbp

def problem2(): 
    from skimage.feature import local_binary_pattern
    prob2img = np.array(Image.open('./face.png').convert('L'))
    lbp = extract_local_binary_pattern(prob2img)
    ref_lbp = local_binary_pattern(prob2img,8,1,method='default')
    print('hi')
    plt.subplot(1,3,1)
    plt.imshow(prob2img)
    plt.title('original')
    plt.subplot(1,3,2)
    plt.imshow(lbp)
    plt.title('Ramon\'s LBP')
    plt.subplot(1,3,3)
    plt.imshow(ref_lbp)
    plt.title('Skimage implementation')
    plt.show()

def problem3():
    from skimage.filters.rank import entropy
    from skimage.morphology import disk 
    from skimage.filters import  gaussian,laplace
    from skimage.feature import local_binary_pattern
    from sklearn.cluster import KMeans
    from skimage import color
    print('hi')
    prob3img = np.array(Image.open('./img3.png').convert('L'))
    img_x,img_y = prob3img.shape
    img_entropy  = entropy(prob3img, disk(25))
    entr_feat= (img_entropy).flatten()
    lbp_feat = local_binary_pattern(prob3img,1,8,'uniform').flatten()
    inten_feat= gaussian(prob3img,sigma=5).flatten()
    edge_map = laplace(prob3img,ksize=10).flatten()
    data_mat = np.vstack([entr_feat,lbp_feat,inten_feat]).T
    model = KMeans(n_clusters=5,n_jobs=-1)
    labels = model.fit_predict(data_mat)    
    out_labels = labels.reshape((img_x,img_y))
    plt.imshow(color.label2rgb(out_labels))
    plt.show()        
def problem1(): 
    num_levels = 2
    img_shape = 3
    #make a dummy image to count and verify 
    print('---my image --')
    img =  (np.random.rand(img_shape,img_shape)*num_levels).astype(np.int32)
    print(img)
    counts = glcm(img,num_levels=num_levels,direction=0)
    print('---counts---')
    print(counts)
def main():
    #problem1()
    #problem2()
    problem3()


if __name__=="__main__":
    print('hello')
    main()

