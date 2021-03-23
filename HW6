import numpy as np
from numpy.core.numeric import outer 
import numpy.typing as nptype 
from typing import NewType,List, Tuple

pixel = NewType('pixel',Tuple[int,int])
contour = NewType('contour',List[pixel])

def abs(x):
    pass 
def grad_func(img,dir): 
    #for the sake of example we'll use the gradient fucntion used in skimage 
    #benefit of this is that the maximum of iis already included 
    pass 


def are_neighbors(i_pix:pixel,j_pix:pixel) -> bool:
    #look to see if your pixels are 1 unit change away from each other 
    #if any is greater than 1 automatically return false
    pass  
def find_dir(i_pix:pixel,j_pix:pixel) -> int : 
    #compare x and y values to establish if we are going left,right,down left
    # we would return a number such as follows 
    # if lr return 0 
    #this is used for indexing 
    pass 
def comp_grads(img:nptype.ArrayLike) -> List[nptype.ArrayLike]:
    #define the gradient neighboroods to be calculate 
    lr = [[-1,0,1],[-1,0,1],[-1,0,1]] 
    rl = -1*lr
    ud = [[1,1,1],[0,0,0],[-1,-1,-1]] #up down 
    du = -1 *  ud   #down up 
    #evaluate each gradient such that we get the gradient map in each orientation 
    grads = [grad_func(img,lr),grad_func(img,rl),grad_func(img,ud),grad_func(img,du)]
    return  grads
def gen_graph(img:nptype.ArrayLike): 
    adj_mat = np.zeros(img.shape) 
    idx_list = list()  #list of pixel locations. such that 10th pixel is a tuple (i,j) representing original img index 
    #represent the image as a flat list of indices 
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]): 
            idx_list.append((i,j))
    # we first need to compute gradients for each of the directs 
    grads = comp_grads(img)
    #iterate through all the pixel rows 
    for i_idx in range(0,len(idx_list)):
        for j_idx in range(0,len(idx_list)):
            pix_i:pixel = idx_list[i_idx]
            pix_j:pixel = idx_list[j_idx]
            if are_neighbors(pix_i,pix_j): 
                dir = find_dir(pix_i,pix_j) 
                adj_mat[i,j] = np.exp(grads[dir][pix_i] + grads[dir][pix_j] ) #define edges using gradient equation from paper
    return (idx_list,adj_mat)

def isInnerContour(i_contour:contour,point:pixel):
    #find all the contour points lying on the same x axis 
    same_ax = [e for e in i_contour if e[0] == point[0] and e[1]]  # find the points sharing the same x axis and to the right 

    if len(same_ax)%2 ==0: 
        # if we  see an even number of points that means we entered and exited the contour therfore  point is outside contour 
        return  False   
    else: 
        # if we only see an odd number of  points to the right we are "inside"  the contour 
        return True 


def dilate_contour(point_list,i_contour,step:int ): 
    i_contour = set(i_contour)
    point_list = set(point_list)
    considered_points = point_list - i_contour # new contour points to be added  
    inner_contour = set() 
    outer_contour = set() 
    for point in considered_points: 
        # calculate distances between  point   and the initial contour points.
        distances_p = np.array([0,10,3,5])  
        isvalid = np.any(distances_p<=step) #check if it  is in valid neighborhood 
        # if any  distance value is less than step size we will add it to the contour set  
        if isvalid:
            if isInnerContour(i_contour,point): 
                inner_contour.add(point) 
            else:
                outer_contour.add(point) 
    bigContour = inner_contour + outer_contour
    pruned_inner_contour = prune_contour_sets(inner_contour,bigContour)
    pruned_outer_contour = prune_contour_sets(outer_contour,bigContour)
    return (pruned_inner_contour,pruned_outer_contour)
def prune_contour_sets(found_contour:contour,bigContour:contour): 
    new_set  = list()
    for point in found_contour:  
        neighbors = [ neighbor in bigContour for neighbor in gen_neighbors(point)  ] 
        if not np.all(neighbors) : # if not all neighbors are in  the giant contour set that means we are in the edge 
            new_set.append(point)
    return new_set


def gen_neighbors(point,point_list) -> List[pixel]: 
    #neighbors are all the points who are 1 unit away.using block distance 
    #return all the valid neighbors of point 
    pass 


def update(idx_list,adj_mat,sink_list,source_list):
    avil_nodes = set(idx_list) - set(sink_list) - set(source_list) 
    source = {(-1,-1):{}}
    new_adj_mat = adj_mat  # this should be acopy of the original adjecency matrix 

    #iterate through the adjacency matrix: 
        # any edge that goes from a  sink node to a source node is set to zero/removed 
    #For each element in the source  list
        #find the neighbors to that node and add them to the source dictionary
        # they should be added such that if point (i,j) is already there it's weight is incremented the appropriate amount 
        # update edge connections from sink to avail_nodes using the source dictionary. 
    #For each node in the sink list: 
        # ge the  sum of the edges connecting to that node (excluding  edges connecting sink to sink )  
        # # update the corespondiing values to equal the sum 

    return adj_mat

def s_t_mincut(new_adj_mat,source_list,sink_list):
    pass  


def checkUnique(old_cont:contour,new_cont:contour):
    #we define a  new contour as not unique if there are less than 10 "new" points 
    
    diff = len(set(old_cont).difference(set(new_cont)))
    if  diff < 10: 
        return False
    else:
        return True 

def  gcbac(img:nptype.ArrayLike,i_contour:contour): 
    (idx_list,adj_mat) = gen_graph(img) 
    isUnique = True 
    step = 0 
    alpha = 5 # this is the step size this will be an arbitrary value for now 
    num_iters =1000
    i =0 
    while i < num_iters :
        sink_list,source_list,merged_cont = dilate_contour(idx_list,i_contour,alpha ) 
        new_adj_mat = update(idx_list,adj_mat,sink_list,source_list)
        n_contour = s_t_mincut(new_adj_mat,source_list,sink_list) 
        isUnique = checkUnique(n_contour,i_contour) 
        if not isUnique:
            return n_contour
        else: 
            i_contour = n_contour

if __name__ == "__main__": 
    #fake main  with fake inputs 
    gcbac('img',[(0,0),(1,1)])
