from PIL import Image
import os
from os.path import abspath
from os.path import join as join
from multiprocessing import process
import random

class Cut_Image():

    '''
    ### Parameters:
        - root:The root path of dataset
        - data_root: The path of file  'rain_filenames.txt'
        - label_root:The path of file 'origin_filenames.txt'        
    '''
    def __init__(self, root, cut_number):
        self.imgs = []
        self.labels = []
        self.cut_number = cut_number
        self.data_path = join(root, 'datasets', 'rain_filenames.txt')
        self.label_path = join(root, 'datasets', 'origin_filenames.txt')

        with open(self.data_path, 'r') as f:            #生成data路径
            self.imgs += [abspath(join(root, 'datasets', 'rain', x.strip())) for x in f.readlines()]

        with open(self.label_path, 'r') as f:           #生成label路径
            self.labels += [abspath(join(root, 'datasets', 'origin', x.strip())) for x in f.readlines()]
        num = 0
        for j in range(self.cut_number):
            a = random.randint(0, 448)  # 生成随机裁剪位置
            b = random.randint(0, 448)

            for i, path in enumerate (self.imgs):

                # print('The current data path is: ' + self.imgs[i])            
                data = Image.open(self.imgs[i])

                # print('The current label path is: ' + self.labels[i])
                label = Image.open(self.labels[i])

                outdata = data.transform(size=(64, 64), method=Image.QUAD, data=(
                    a, b, a+64, b, a, b+64, a+64, b+64))   #裁剪data

                outlabel = label.transform(size=(64, 64), method=Image.QUAD, data=(
                    a, b, a + 64, b, a, b + 64, a + 64, b + 64))  # 裁剪label

                newname = self.imgs[i].split('.')[0].split('\\')[-1] + '_' + str(j + 1) + '.jpg' #新文件名
                newdir = self.labels[i].split('.')[0].split('\\')[-2]
                

                new_datapath = os.path.join(
                    root, 'crop_datasets', 'mini_data', newdir, newname)  # 裁剪data结果输出路径
                print("The new datapath is :" + new_datapath)
                outdata.save(new_datapath)  #保存新data

                new_labelpath = os.path.join(
                    root, 'crop_datasets', 'mini_label', newdir, newname)  # 裁剪label结果输出路径
                print("The new labelpath is :" + new_labelpath)
                outlabel.save(new_labelpath)  #保存新label
                print(num)
                num += 1
            

if __name__ == '__main__':
    for i in range(8):
        P = process(target=Cut_Image, args=(r"C:\Users\Adam\Desktop", 3))
        P.start()
        P.join()
