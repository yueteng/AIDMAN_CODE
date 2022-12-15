import os
for root, dirs, files in os.walk("D:\Tuo\yolov5-cell\inference\output/"):
    print(files) #当前路径下所有非目录子文件

for file in files:
    with open ("D:\Tuo\yolov5-cell\inference\output/"+file,"r") as f:
        lines=f.readlines()
        print(len(lines))