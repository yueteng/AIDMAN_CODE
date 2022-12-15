import csv
import os
import pandas as pd

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
#测试样例1


for root, dirs, files in os.walk("D:\Tuo\yolov5-cell\csv/"):
    print(files) #当前路径下所有非目录子文件

csvname = "D:\Tuo\yolov5-cell\csv/" + "result.csv"
csvfile = open(csvname, 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(csvfile)
csv_writer.writerow(["name", "gt","pre","IOU0.5", "IOU0.75"])

for file in files:
    count5=0
    count75=0
    if file[0]=="g":
        data1 = pd.read_csv("D:\Tuo\yolov5-cell\csv/"+file)
        file2="dr"+file[2:]
        data2 = pd.read_csv("D:\Tuo\yolov5-cell\csv/"+file2)
        for box_num1 in range(len(data1)):
            box1=data1.loc[box_num1]
            for box_num2 in range(len(data2)):
                box2 = data2.loc[box_num2]
                IOU = compute_IOU(box1, box2)
                if IOU>=0.5:
                    count5+=1
                if IOU>=0.75:
                    count75+=1
        csv_writer.writerow([file[2:],len(data1),len(data2), count5, count75])
        print("file:"+file[2:]+" gt:"+str(len(data1))+" pre:"+str(len(data2))+" IOU0.5:"+str(count5)+" Iou0.75:"+str(count75))