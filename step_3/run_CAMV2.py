import os
import glob


dt="Negtivate"
gpu="1"

files=os.listdir( "train_data_20220824/step_2/"+dt+"/")

for file in files:
    if os.path.exists("D:/YS/M_D/Updata_2/evaluation/train_data_20220824/CAM/"+dt+"/"+file):
        print("continue",end=" ")
        print(file)
        continue
    else:
        try:
            os.mkdir("D:/YS/M_D/Updata_2/evaluation/train_data_20220824/CAM/"+dt+"/"+file)
        except:
            continue
    print("\n\n\n\n\n\n")
    print("train_data_20220824/step_2/"+dt+"/"+file)
    os.system("python CAM_V2.py train_data_20220824/step_2/"+dt+"/"+file+"/ "+gpu)
    print()