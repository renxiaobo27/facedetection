import tensorflow as tf
import vgg
import numpy as np
from util import get_img


import math
import matplotlib.pyplot as plt
#visulize layer
def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.plot()

#feature extraction based on pre-trained model
if __name__ =='__main__':
    img_path='t.jpg'
    vgg_model_path = 'imagenet-vgg-verydeep-19.mat'
    img = get_img('t.jpg',(256,256,3)).astype(np.float32)
    shape = (1,) + img.shape


    select_features = {}

    Select_Layers =('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    with tf.Graph().as_default(),tf.device('/cpu:0'),tf.Session() as sess:
        image = tf.placeholder(tf.float32, shape=shape, name='style_image')
        style_image_pre = vgg.preprocess(image)
        net_vgg = vgg.net(vgg_model_path, style_image_pre)

        img = np.array([img])
        for layer in Select_Layers:
            features = net_vgg[layer].eval(feed_dict={image: img})
            plotNNFilter(features)
            features = np.reshape(features, (-1, features.shape[3]))
            select_features[layer] = features
