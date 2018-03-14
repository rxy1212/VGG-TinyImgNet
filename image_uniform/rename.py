import os 
import cv2
from glob import glob

path = []
for fn in glob(r".\datasets_origin\*\*\*.jpg"):
    path.append(fn)
    with open('./pathname.txt', 'a') as f:
        f.write(fn)
        f.write('\n')
for i in range(6):
    for num in range(2500*i, 2500*(i+1)):
        img = cv2.imread(path[num])
        cv2.imwrite('.\\datasets_origin\\batch' + str(i) + '\\' + 'batch' + str(i) + '_' +  str(num%2500) + '.jpg', img)
        num += 1
        print('imagenum:', num)
for num in range(2500*6, len(path)):
    img = cv2.imread(path[num])
    cv2.imwrite('.\\datasets_origin\\batch6\\' + 'batch6' + '_' + str(num%2500) + '.jpg', img)
    num += 1
    print('imagenum:', num)
print('totalnum:', num)