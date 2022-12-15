import cv2
import os
import numpy as np
xml_head = '''<?xml version="1.0"?>
<annotation>
    <folder>VOC2007</folder>
    <filename>{}</filename>
    <path>{}</path>
    <source>
        <database>unknow</database>
    </source>  
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''
xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Unspecified</pose>
        <!--是否被裁减，0表示完整，1表示不完整-->
        <truncated>0</truncated>
        <!--是否容易识别，0表示容易，1表示困难-->
        <difficult>0</difficult>
        <!--bounding box的四个坐标-->
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''

path  = 'D:\Tuo\yolov5-cell\detect\output/'
cnt = 0
for yolo_file in os.listdir(path):
    # jpg = 'images/' + yolo_file.replace('.txt', '.jpg')  # image path
    # txt = 'labels/' + yolo_file  # yolo label txt path
    # xml_path = 'Annotations/' + yolo_file.replace('.txt', '.xml')  # xml save path

    if yolo_file.endswith('.txt'):
        jpg = path + yolo_file.replace('.txt', '.jpg')
        txt = path + yolo_file
        xml_path = path + yolo_file.replace('.txt', '.xml')
        obj = ''
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), 1)
        img_h, img_w = img.shape[0], img.shape[1]
        head = xml_head.format(str(jpg),str(yolo_file.replace('.txt', '.bmp')), str(img_w), str(img_h), str(1))
        with open(txt, 'r') as f:
            for line in f.readlines():
                yolo_datas = line.strip().split(' ')
                #label = int(float(yolo_datas[0].strip()))
                label = str(yolo_datas[0].strip())

                center_x = round(float(str(yolo_datas[1]).strip()) * img_w)
                center_y = round(float(str(yolo_datas[2]).strip()) * img_h)
                bbox_width = round(float(str(yolo_datas[3]).strip()) * img_w)
                bbox_height = round(float(str(yolo_datas[4]).strip()) * img_h)

                # center_x = round(float(str(yolo_datas[2]).strip()))
                # center_y = round(float(str(yolo_datas[3]).strip()))
                # bbox_width = round(float(str(yolo_datas[4]).strip()))
                # bbox_height = round(float(str(yolo_datas[5]).strip()))
                '''
                xmin = str(int(center_x - bbox_width / 2))
                ymin = str(int(center_y - bbox_height / 2))
                xmax = str(int(center_x + bbox_width / 2))
                ymax = str(int(center_y + bbox_height / 2))
                '''
                xmin = str(center_x)
                ymin = str(center_y)
                xmax = str(bbox_width)
                ymax = str(bbox_height)

                obj += xml_obj.format(label, xmin, ymin, xmax, ymax)
        with open(xml_path, 'w',encoding='utf-8') as f_xml:
            f_xml.write(head + obj + xml_end)
        cnt += 1
        print(cnt)

