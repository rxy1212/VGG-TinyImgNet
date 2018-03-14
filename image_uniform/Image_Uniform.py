import cv2
import os
from os.path import dirname, join
from glob import glob

def main():
    total_number = 0
    for path, dirnames, files in os.walk('./Imagedata'):
        print(path, dirnames)
        for dirs in dirnames:
            if not os.path.exists('./Uniformed_data/' + dirs):
                os.mkdir('./Uniformed_data/' + dirs)
            num = 0
            for fn in glob(join('./Imagedata/' + dirs + '/', '*.jpg')):
                img = cv2.imread(fn)
                if(img is not None):
                    res = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(r'./Uniformed_data/' + dirs + '/' +str(num) + '.jpg', res)
                    img_flip = cv2.flip(res, 1)           
                    cv2.imwrite(r'./Uniformed_data/' + dirs + '/' + str(num) + '_1' + '.jpg', img_flip)
                    print("images_num:", num)
                    num += 1
                    total_number += 1
    print('total_number:', total_number)
    print('All images were uniformed !')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()