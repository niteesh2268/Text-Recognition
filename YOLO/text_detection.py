
from keras.models import Model
from keras.models import model_from_json
#json_file = open('text_detect_model.json', 'r')
#model_json = json_file.read()
#model = model_from_json(model_json)


#model.load_weights('text_detect.h5')

def decode_to_boxes(output , ht , wd):
    #output : (x,x,1,5)
    #x,y,h,w

    img_ht = ht
    img_wd = wd
    threshold = 0.5
    grid_h,grid_w = output.shape[:2]
    final_boxes = []
    scores = []

    for i in range(grid_h):
        for j in range(grid_w):
            if output[i,j,0,0] > threshold:

                temp = output[i,j,0,1:5]
                
                x_unit = ((j + (temp[0]))/grid_w)*img_wd
                y_unit = ((i + (temp[1]))/grid_h)*img_ht
                width = temp[2]*img_wd*1.3
                height = temp[3]*img_ht*1.3
                
                final_boxes.append([x_unit - width/2,y_unit - height/2 ,x_unit + width/2,y_unit + height/2])
                scores.append(output[i,j,0,0])
    
    return final_boxes,scores



def iou(box1,box2):

    x1 = max(box1[0],box2[0])
    x2 = min(box1[2],box2[2])
    y1 = max(box1[1] ,box2[1])
    y2 = min(box1[3],box2[3])
    
    inter = (x2 - x1)*(y2 - y1)
    
    area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area2 = (box2[2] - box2[0])*(box2[3] - box2[1])
    fin_area = area1 + area2 - inter
        
    iou = inter/fin_area
    
    return iou



def non_max(boxes , scores , iou_num):

    scores_sort = scores.argsort().tolist()
    keep = []
    
    while(len(scores_sort)):
        
        index = scores_sort.pop()
        keep.append(index)
        
        if(len(scores_sort) == 0):
            break
    
        iou_res = []
    
        for i in scores_sort:
            iou_res.append(iou(boxes[index] , boxes[i]))
        
        iou_res = np.array(iou_res)
        filtered_indexes = set((iou_res > iou_num).nonzero()[0])

        scores_sort = [v for (i,v) in enumerate(scores_sort) if i not in filtered_indexes]
    
    final = []
    
    for i in keep:
        final.append(boxes[i])
    
    return final


def decode(output , ht , wd , iou):
    
    
    boxes , scores = decode_to_boxes(output ,ht ,wd)
    
    
    boxes = non_max(boxes,np.array(scores) , iou)
    
    
    return boxes

import matplotlib.pyplot as plt
import cv2
from PIL import Image
def predict_func(model , inp , iou,num):

    ans = model.predict(inp)
    
    #np.save('Results/ans.npy',ans)
    boxes = decode(ans[0] , 512 , 512 , iou)
    #print(ans)
    img = ((inp + 1)/2)
    img = img[0]
    #plt.imshow(img)
    #plt.show()

    
    for i in boxes:
        num=num+1
        i = [int(x) for x in i]

        img = cv2.rectangle(img , (i[0] ,i[1]) , (i[2] , i[3]) , color = (0,255,0) , thickness = 2)
        img_temp=img
        #print(i)
        for k in range(0,4):
          i[k]=max(i[k],0)
        for k in range(0,4):
          i[k]=min(i[k],512)  
        crop_img = img[i[1]:i[3], i[0]:i[2]]
        plt.imshow(crop_img)
        #plt.show()
        plt.savefig('./output/'+str(num))
    #plt.imshow(img)
    #plt.show()
    return num
    
import os
files=os.listdir('./images')
print(files)
num=0
import numpy as np
for imgx in files:
    test_image=cv2.imread('./images/'+imgx)   
    img = cv2.resize(test_image,(512,512))
    list_images=[]
    list_images.append(img)
    list_images=np.array(list_images)
    list_images = (list_images - 127.5)/127.5
    num=predict_func(model,list_images[0:1],0.5,num)


