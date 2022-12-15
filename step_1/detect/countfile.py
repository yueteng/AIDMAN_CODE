'''
    #python将某文件夹下的文件名存储到excel中
'''

#导入所需模块
import os
import xlwt

#定义要处理的文件路径（文件夹）
file_dir = "D:\Tuo\yolov5-cell\detect/Positivate_sp"

#将文件名列出并存储在allfilenames里面
allfilenames = os.listdir(file_dir)
#打印看是否符合预期
print(allfilenames)

#创建工作簿
workbook = xlwt.Workbook()
#新建工作表，并且命名为fileNames
worksheet = workbook.add_sheet('fileNames')
worksheet.write(0, 0, "pic_name")
worksheet.write(0, 1, "pic_number")
#开始往表格里写文件名
n = 1 #定义起始行数
for i in allfilenames:
    if i[-3:] !="jpg" and i[-3:] !="png" and i[-3:]!="xls":
        worksheet.write(n, 0, i) #向单元格里内写入i
        worksheet.write(n, 1, str(len(os.listdir(file_dir+"/"+i))))

        n += 1 #写完一个i写完一行后n自加1

#保存工作簿
workbook.save(file_dir+'/count.xls')