from numpy import load
import pandas as pd 
from PIL import Image
import pdb 
import numpy as np 
def yolo2voc(sub, img_name):
    img = np.array(Image.open(img_name))
    (h,w,d) = img.shape
    s = [w,h];
    for i in range(0,sub.shape[0]):
        xmin = sub.bx.iloc[i] - sub.bw.iloc[i]/2;
        xmax = sub.bx.iloc[i] + sub.bw.iloc[i]/2;
        ymin = sub.by.iloc[i] - sub.bh.iloc[i]/2;
        ymax = sub.by.iloc[i] + sub.bh.iloc[i]/2;
        box= [xmin,xmax,ymin,ymax]; 
        (bx,by,bw,bh) = convert(s,box); 
        sub.bx.iloc[i] = bx 
        sub.by.iloc[i] = by 
        sub.bw.iloc[i] = bw 
        sub.bh.iloc[i] = bh 
    return sub 

def  convert(s,box): 
    dw = 1./(s[0])
    dh = 1./(s[1]) 
    x = (box[0] + box[1])/2.0 - 1 
    y = (box[2] + box[3])/2.0 - 1;
    w = box[1]- box[0]
    h = box[3] - box[2]
    x = x*dw;
    w = w*dw;
    y = y*dh;
    h = h*dh;
    return (x,y,w,h)

def convert_back(df,s):  
    bbx = pd.DataFrame()
    dw = s[0]
    dh = s[1]  
    cx = df.bx*dw
    cy = df.by*dh 
    h  = df.bh*dh
    w = df.bw*dw 
    bbx['Name'] =df.Name
    bbx['xmin'] = cx - w/2
    bbx['xmax']= cx +w/2 
    bbx['ymin'] = cx + h/2
    bbx['ymax'] = cy  + h/2 
    if 'Prob' in df.keys(): 
        bbx['Prob']= df.Prob
    if 'im_name' in df.keys():
        bbx['im_name']=df.im_name 
    return bbx


def find_interesect(bbx_gts, bbx_pred):
        #reminder bbx arrays are formated as 
        # xmin,xmax,ymin,ymax
        #find the areas shared by both. out vars will be a combiantion of both 
        xmin = np.maximum(bbx_gts[:, 0], bbx_pred[0])
        xmax = np.minimum(bbx_gts[:, 1], bbx_pred[1])
        ymin = np.maximum(bbx_gts[:, 2], bbx_pred[2]) 
        ymax = np.minimum(bbx_gts[:, 3], bbx_pred[3])
        iw = np.maximum(xmax-  xmin  + 1., 0.)
        ih = np.maximum(ymax-ymin + 1., 0.)
        return iw*ih 

def find_union(bbx_gts, bbx_pred,intersect): 
    prd_area = (bbx_pred[1] - bbx_pred[0] + 1) * (bbx_pred[3] - bbx_pred[2] + 1) 
    gt_area = (bbx_gts[:, 1] - bbx_gts[:, 0] + 1) * (bbx_gts[:, 3] - bbx_gts[:, 2] + 1)
    union = prd_area + gt_area   - intersect
    return  union 

def calc_iou(label,pred,img_name):
    img = np.array(Image.open(img_name))
    (h,w,d) = img.shape 
    #label = label[label['Name']==class_interest]
    #pred = pred[pred['Name'] == class_interest]
    pred = pred.sort_values('Prob',axis=0,ascending=False)
    pred = convert_back(pred,(w,h))
    label = convert_back(label,(w,h)) 
    fps = np.zeros((pred.shape[0],))
    tps = np.zeros((pred.shape[0],))
    #iterate through the predicitons 
    bbx_gts= label[['xmin','xmax','ymin','ymax']].to_numpy()  
    for i in range(0,pred.shape[0]):
        bb =  pred.iloc[i][['xmin','xmax','ymin','ymax'] ].to_numpy()
        intersection = find_interesect(bbx_gts,bb)
        union = find_union(bbx_gts,bb,intersection )
        overlaps = intersection/union
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ovmax >= .5 : 
            if label.iloc[jmax]['Name']  == pred.iloc[i]['Name']:
                tps[i]= 1
            else: 
                fps[i]
        else: 
            fps[i] = 1
    return tps,fps
def calc_class_AP(cl):
    my_data = data.copy()
    print(cl)
    my_data= my_data[my_data['Name']==cl] 
    img_names = my_data['im_name'].unique()
    tp_list = list()
    fp_list = list() 
    counter = 0
    for  e in img_names: 
        try: 
            txt_path =  "{}{}.txt".format(annot_paths,e)
            label = pd.read_csv(txt_path,header=None,names=["Name", "bx", "by", "bw", "bh"],sep=' ')
            model_out =  my_data[my_data['im_name']==e]
            img_name = '{}/{}.jpg'.format(jpg_path,model_out['im_name'].iloc[0])
            outs  = yolo2voc(model_out,img_name) 
            tp,fp = calc_iou(label,outs,img_name)
            tp_list.append(tp)
            fp_list.append(fp)
        except FileNotFoundError: 
            print('skipping due to read error ')
            continue 
    tp = np.cumsum(np.hstack(tp_list) )
    fp = np.cumsum( np.hstack(fp_list)) 
    prec = np.mean(tp/(tp+fp))
    rec = tp/ my_data.shape[0]
    mrec = np.concatenate(([0.], rec, [1.]))
    prec = tp/(tp+fp)
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap 
def eval_bbox(og_data,annot_paths,jpeg_path):
    #filter by class  
    classes = og_data['Name'].unique()
    ap_list = list() 
    for name in classes: 
        data = og_data[og_data['Name']==name]
        img_names = data['im_name'].unique()
        annot_paths = './VOCdevkit/VOC2012/labels/';
        tp_list = list()
        fp_list = list() 
        for  e in img_names: 
            try: 
                txt_path =  "{}{}.txt".format(annot_paths,e)
                label = pd.read_csv(txt_path,header=None,names=["Name", "bx", "by", "bw", "bh"],sep=' ')
                #lets find the predicitons regaridng an image 
                model_out =  data[data['im_name']==e]
                #find the image 
                img_name = '{}/{}.jpg'.format(jpeg_path,model_out['im_name'].iloc[0])
                outs  = yolo2voc(model_out,img_name) 
                tp,fp = calc_iou(label,outs,img_name)
                tp_list.append(tp)
                fp_list.append(fp)
            except FileNotFoundError: 
                print('skipping due to read error ')
                continue 
        tp = np.cumsum(np.hstack(tp_list) )
        fp = np.cumsum( np.hstack(fp_list)) 
        prec = tp/(tp+fp)
        rec = tp/ data.shape[0]  # divide by total number of positive cases 
        #going to calculate area under curve 
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        #iterate in decresing order 
        #gen interpolated buckets
    buckets = np.array_split(prec,11)
    aps = np.zeros((11,))
    for i in range(0,11): 
        aps[i] = np.mean(buckets[i]) 
    ap =np.mean(aps) 
    ap_list.append(ap) 
    return np.mean(ap_list)

if __name__=="__main__": 
    model_comp ='tiny'
    og_data  = pd.read_csv('./ramon_outputs/outs.csv')
    annot_paths = './VOCdevkit/VOC2012/labels/'
    jpeg_data = "./VOCdevkit/VOC2012/JPEGImages"
    Map = eval_bbox(og_data,annot_paths,jpeg_data)
    print('hi')
    pdb.set_trace() 
