# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:39:17 2020

@author: nihao
"""
import os
import numpy as np
import codecs
import json
import glob
import cv2
import shutil
#from sklearn.model_selection import train_test_splitimport os
import numpy as np
import codecs
import json
import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split

# 1.标签路径
labelme_path = "LabelmeData_test"  # 原始labelme标注数据路径
saved_path = "voctest"  # 保存路径

# 2.创建要求文件夹
dst_annotation_dir = os.path.join(saved_path, 'Annotations')
if not os.path.exists(dst_annotation_dir):
    os.makedirs(dst_annotation_dir)
dst_image_dir = os.path.join(saved_path, "JPEGImages")
if not os.path.exists(dst_image_dir):
    os.makedirs(dst_image_dir)
dst_main_dir = os.path.join(saved_path, "ImageSets", "Main")
if not os.path.exists(dst_main_dir):
    os.makedirs(dst_main_dir)

# 3.获取待处理文件
org_json_files = sorted(glob.glob(os.path.join(labelme_path, '*.json')))
org_json_file_names = [i.split("\\")[-1].split(".json")[0] for i in org_json_files]
org_img_files = sorted(glob.glob(os.path.join(labelme_path, '*.jpg')))
org_img_file_names = [i.split("\\")[-1].split(".jpg")[0] for i in org_img_files]

# 4.labelme file to voc dataset
for i, json_file_ in enumerate(org_json_files):
    json_file = json.load(open(json_file_, "r", encoding="utf-8"))
    image_path = os.path.join(labelme_path, org_json_file_names[i]+'.jpg')
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    dst_image_path = os.path.join(dst_image_dir, json_file_.split("\\")[-1].split(".")[0]+".jpg".format(i))
    cv2.imwrite(dst_image_path, img)
    dst_annotation_path = os.path.join(dst_annotation_dir, '{:06d}.xml'.format(i))
    with codecs.open(dst_annotation_path, "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'Pin_detection' + '</folder>\n')
        xml.write('\t<filename>' + json_file_.split("\\")[-1].split(".")[0]+".jpg" + '</filename>\n')
        # xml.write('\t<source>\n')
        # xml.write('\t\t<database>The UAV autolanding</database>\n')
        # xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        # xml.write('\t\t<image>flickr</image>\n')
        # xml.write('\t\t<flickrid>NULL</flickrid>\n')
        # xml.write('\t</source>\n')
        # xml.write('\t<owner>\n')
        # xml.write('\t\t<flickrid>NULL</flickrid>\n')
        # xml.write('\t\t<name>ChaojieZhu</name>\n')
        # xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + label + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_file_, xmin, ymin, xmax, ymax, label)
        xml.write('</annotation>')
    os.rename(dst_annotation_path,"voctest\\Annotations\\"+json_file_.split("\\")[-1].split(".")[0]+".xml")



# 5.split files for txt
train_file = os.path.join(dst_main_dir, 'train.txt')
trainval_file = os.path.join(dst_main_dir, 'trainval.txt')
val_file = os.path.join(dst_main_dir, 'val.txt')
test_file = os.path.join(dst_main_dir, 'test.txt')

ftrain = open(train_file, 'w')
ftrainval = open(trainval_file, 'w')
fval = open(val_file, 'w')
ftest = open(test_file, 'w')

total_annotation_files = glob.glob(os.path.join(dst_annotation_dir, "*.xml"))
total_annotation_names = [i.split("\\")[-1].split(".xml")[0] for i in total_annotation_files]

test_filepath = "VOC2007_\ImageSets\Main"
for file in total_annotation_names:
    ftrainval.writelines(file + '\n')
# test
for file in os.listdir(test_filepath):
   ftest.write(file.split(".jpg")[0] + "\n")
# split
train_files, val_files = train_test_split(total_annotation_names, test_size=0.2)
# train
for file in train_files:
    ftrain.write(file + '\n')
# val
for file in val_files:
    fval.write(file + '\n')

ftrainval.close()
ftrain.close()
fval.close()
# ftest.close()