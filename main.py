# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/09/02 14:32:39
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   测试辅助标记的结果
'''
import cv2
from watermarkers import water_markers
from buildjson import BuildJson
import numpy as np

# def main(image_path,markers):
    
#     # 读取原始图片


if __name__ == '__main__':
    image_path='./demo_image.png'
    markers=cv2.imread('./demo_markers.png')
    print(np.unique(markers[...,0]))

    save_json_path='./demo_image.json'

    labels_mask,labels_name=water_markers(image_path,markers[...,0])
    
    buildjson=BuildJson()
    buildjson.svae_mask_to_json(image_path,labels_mask,labels_name,save_json_path)
