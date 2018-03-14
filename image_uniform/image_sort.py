import os 
import cv2
from glob import glob

parts = ['./part1', './part2', './part3', './part4']
total_num = 0
for i in range(4):
    for path, dirnames, _ in os.walk(parts[i]):
        for dirs in dirnames:
            new_path = './new/' + 'part' + str(i+1) + '/' + dirs + '/'
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            num = 0
            for root, _, filenames in os.walk(parts[i] + '/' + dirs):
                filenames.sort(key=lambda d: int(d.split(".")[0].split("_")[0]))
            for file in filenames:
                fn = root + '/' + file
                img  = cv2.imread(fn)
                # if img is not None:
                if(num % 2 == 0): 
                    cv2.imwrite(new_path + str(num//2) + '.jpg', img)
                elif(num % 2 ==1):
                    cv2.imwrite(new_path + str(num//2) + '_1' + '.jpg', img)
                else:
                    print("div error!")
                num += 1
                total_num += 1
                print('imagenum:', num)

print('total_num:', total_num)
print("All images were sorted! ")
cv2.waitKey(0)
cv2.destroyAllWindows()

