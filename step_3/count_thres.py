import numpy as np
import pandas as pd
import os

data_dir="D:/YS/M_D/Updata_2/evaluation/train_data_20220824/step_2/"
Negtivate_files=os.listdir(data_dir+"Negtivate/")
Negtivate=[]
for n in Negtivate_files:
    Negtivate.append(list(pd.read_csv(data_dir+"Negtivate/"+n+"/"+n+"_cell.csv")["pred"]))

Positivate_files=os.listdir(data_dir+"Positivate/")
Positivate=[]
for n in Positivate_files:
    Positivate.append(list(pd.read_csv(data_dir+"Positivate/"+n+"/"+n+"_cell.csv")["pred"]))
f=open("D:/YS/M_D/Updata_2/evaluation/train_data_20220824/step_2/thres.csv","w")
FP=0
FN=0
f.write("Thres,acc,FP,FN\n")
for thres in range(63):
    thres+=1
    acc=[]
    for p in Positivate:
        for i in range(len(p)):
            if p[i]<0.5:
                p[i]=0
            else:
                p[i]=1
        a=sum(p)
        # print(a,end=" ")
        if a>=thres:
            acc.append(1)
        else:
            FN+=1
            acc.append(0)
    for p in Negtivate:
        for i in range(len(p)):
            if p[i]<0.5:
                p[i]=0
            else:
                p[i]=1
        a=sum(p)
        # print(a,end=" ")
        if a<thres:
            acc.append(1)
        else:
            FP+=1
            acc.append(0)

    print(thres)
    print(sum(acc)/len(acc))

    f.write(str(thres)+","+str(sum(acc)/len(acc))+","+str(FP/len(Negtivate))+","+str(FN/len(Positivate))+"\n")
