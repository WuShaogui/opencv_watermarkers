# -*- encoding: utf-8 -*-
'''
@File    :   waterMarkers.py
@Time    :   2021/09/02 14:29:55
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   基于opencv的分水岭算法，开发辅助标记功能
'''
import cv2
import numpy as np

def water_markers(image_path,markers):
    # 对markers的判断
    labels=np.unique(markers)
    if(len(markers)==1):
        print("无效标记")
        return None
    else:
        if np.any(markers==0) and np.any(markers==255):
            #print('合法输入')
            image=cv2.imread(image_path)
            markers = cv2.watershed(image.astype('uint8'),markers.astype('int32'))

            labels_mask=[]
            labels_name=[]
            for label in labels:
                if label!=0 and label!=255:
                    label_mask=np.zeros((image.shape[0],image.shape[1]))
                    label_mask[markers ==label]=255
                    labels_mask.append(label_mask)
                    labels_name.append(label)
                    #cv2.imwrite("label_mask-{}.png".format(label),label_mask)
            
            return labels_mask,labels_name
        else:
            print('不存在背景或背景未被标注')
            return None
