import cv2
import numpy as np
import os
import util
class Features():
    '''
    this class provides feature extraction on cv:Mat
    '''
    def __init__(self):
        self.num_bits = 1600
    #load index
        self.index = np.load('index.npy')


    def get_feature(self,image):
        return self._get_bits_feature(self._get_blocks(image))

    def _get_blocks_resize(self,image):
        resized_image = cv2.resize(image,(10,10))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        b = gray_image.reshape(-1)
        return b.tolist()

    def _get_blocks(self,image):
        #assume image color image

        block_step = 10
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows,cols= gray_image.shape
        block=[0]*100
        row_step = rows / block_step
        col_step = cols / block_step

        for i in range(rows):
            for j in range(cols):
                id = (i / row_step) * block_step + (j/col_step)
                if id >=100:
                    continue
                block[id] += gray_image[i][j]
        # print block
        # cv2.imshow('dfd',gray_image)
        # cv2.waitKey(0)
        return [x/100 for x in block]

    def _get_bits_feature(self,block):
        feature = np.ones(self.num_bits, dtype=np.int32)
        id = 0
        for x in (self.index):
            b = x-1
            if b<0:
                b=0
            if block[b] < block[x]:
                feature[id] = 0
            id+=1

        return feature

    @staticmethod
    def hamming_distance(f1,f2):
        return np.count_nonzero(f1 != f2)












def test_shot_segmentation():
    f = Features()
    img_folder = '/home/tcl-admin/data/cartoon_img/sea1'
    img_list = os.listdir(img_folder)
    img_list.sort()
    pre = []

    #rough segment
    for file_name in img_list:
        img_path = os.path.join(img_folder,file_name)
        img = cv2.imread(img_path)
        if len(pre)==0:
            pre = f.get_feature(img)
        else:
            cur = f.get_feature(img)
            dist = f.hamming_distance(pre,cur)
            if(dist>300):
                print img_path
            pre = cur


# img_folder
img_folder = "/home/tcl-admin/data/cartoon_img"
folder_name = 'sea1'

list = util.get_range_image(img_folder,folder_name,141,540)
print len(list)
x_train=[]
ct = 1
for img_path in list:
    img = cv2.imread(img_path)
    x_train.append(img.reshape(-1))



newW=640;newH=480;

from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal
pca = PCA(n_components=30)
pca.fit(x_train)
X_train_pca = pca.transform(x_train)
X_projected = pca.inverse_transform(X_train_pca)

a = X_projected[0]
b = a.reshape(newH,newW,3)
cv2.imwrite('bbb.jpg',b)
cv2.imshow('kkk',b)

print np.sum(X_projected[0] - x_train[0])
cv2.waitKey(0)



