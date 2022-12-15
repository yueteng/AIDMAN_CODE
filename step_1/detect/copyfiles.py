import os
import shutil
i=0
for root,dir1,filename in os.walk("D:\Tuo\yolov5-cell\detect/files/"):
    for index in range(len(filename)):
        if os.path.splitext(filename[index])[1]=='.jpg':#这里注意filename是个元组，splitext方法的时候只能是字符串
            i+=1
            root1="D:\Tuo\yolov5-cell\detect\input/"
            old_path = os.path.join(root, filename[index])
            new_path = os.path.join(root1,filename[index])
            shutil.copyfile(old_path,new_path)

print("总共有",i,"图层文件被复制！")
