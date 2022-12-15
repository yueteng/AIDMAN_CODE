import shutil
import os

file_List = ["train", "val"]
for file in file_List:
    if os.path.exists('../VOC/images/%s' % file):
        shutil.rmtree('../VOC/images/%s' % file)
    os.makedirs('../VOC/images/%s' % file)
    if os.path.exists('../VOC/labels/%s' % file):
        shutil.rmtree('../VOC/labels/%s' % file)
    os.makedirs('../VOC/labels/%s' % file)
    print(os.path.exists('../tmp/%s.txt' % file))
    f = open('../tmp/%s.txt' % file, 'r')
    if file=="test":
        f=f = open('../tmp_test/%s.txt' % file, 'r')
    lines = f.readlines()
    for line in lines:
        print(line)
        line = "/".join(line.split('/')[-5:]).strip()
        shutil.copy(line, "../VOC/images/%s" % file)
        line = line.replace('JPEGImages', 'labels')
        line = line.replace('jpg', 'txt')
        shutil.copy(line, "../VOC/labels/%s/" % file)
