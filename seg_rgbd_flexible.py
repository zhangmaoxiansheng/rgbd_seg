#!/usr/bin/env python
from __future__ import print_function
import os
from os.path import splitext, join, isfile, isdir, basename
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json, load_model
import tensorflow as tf
import layers_builder as layers
from glob import glob
from python_utils import utils
from python_utils.preprocessing import preprocess_img
from keras.utils.generic_utils import CustomObjectScope
import scipy.io
import cv2 as cv
from sklearn.cluster import KMeans

# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape
        json_path = join("weights", "keras", weights + ".json")
        h5_path = join("weights", "keras", weights + ".h5")
        if 'pspnet' in weights:
            if os.path.isfile(json_path) and os.path.isfile(h5_path):
                print("Keras model & weights found, loading...")
                with CustomObjectScope({'Interp': layers.Interp}):
                    with open(json_path, 'r') as file_handle:
                        self.model = model_from_json(file_handle.read())
                self.model.load_weights(h5_path)
            else:
                print("No Keras model & weights found, import from npy weights.")
                self.model = layers.build_pspnet(nb_classes=nb_classes,
                                                 resnet_layers=resnet_layers,
                                                 input_shape=self.input_shape)
                self.set_npy_weights(weights)
        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def predict(self, img, flip_evaluation=False):
        """
        Predict segementation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        h_ori, w_ori = img.shape[:2]

        # Preprocess
        img = misc.imresize(img, self.input_shape)

        img = img - DATA_MEAN
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype('float32')
        print("Predicting...")

        probs = self.feed_forward(img, flip_evaluation)

        if img.shape[0:1] != self.input_shape:  # upscale prediction if necessary
            h, w = probs.shape[:2]
            probs = ndimage.zoom(probs, (1. * h_ori / h, 1. * w_ori / w, 1.),
                                 order=1, prefilter=False)

        print("Finished prediction...")

        return probs

    def feed_forward(self, data, flip_evaluation=False):
        assert data.shape == (self.input_shape[0], self.input_shape[1], 3)

        if flip_evaluation:
            print("Predict flipped")
            input_with_flipped = np.array(
                [data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[
                          0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = join("weights", "npy", weights_path + ".npy")
        json_path = join("weights", "keras", weights_path + ".json")
        h5_path = join("weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()][
                    'mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()][
                    'variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()][
                    'scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()][
                    'offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                  biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)

def using_depth_refined(img,thres = 215):
    #large than thres, smaller than 50 = 0
    ret,thresh1 = cv.threshold(img,thres,255,cv.THRESH_TOZERO_INV)
    ret2,thresh2 = cv.threshold(thresh1,50,255,cv.THRESH_BINARY)
    thresh3 = cv.resize(thresh2,(768,636))
    dilated_res = cv.dilate(thresh3,(5,5))

    return dilated_res

def test_res(cm,mask):
    cm = cm.astype(np.uint16)
    cm1 = cm.astype(np.uint8)
    contours, hierarchy = cv.findContours(cm1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    index = 0
    if len(contours) == 0:
        #print("refine died")
        return cm1,0

    for i in range(0,len(contours)):
        area = cv.contourArea(contours[i])
        if(area > max_area):
            max_area = area
            index = i
    final_contours = contours[index]
    x, y, w, h = cv.boundingRect(final_contours)
    
    equal_ = cm[y:y+h,x:x+w] != mask[y:y+h,x:x+w]
    r = equal_.sum()
    #r = r - np.sum(mask[y:y+h,x:x+w] == 0)
    r = float(r)/(h*w)
    if(r > 0.25):
        #print("start to refine")
        head = int(h/8)
        h = int(h-h/3)
        d_w = int(0.2*w)
        if(x-d_w <= 0):
            d_w = x
        if(x+w+d_w >= cm.shape[1]):
            d_w = cm.shape[1] - 1 - x -w
        if(y-head <0):
            head = 0

        cm[y-head:y+h,x-d_w:x+w+d_w] = mask[y-head:y+h,x-d_w:x+w+d_w] + cm[y-head:y+h,x-d_w:x+w+d_w]
        #cv.rectangle(cm1,(x-d_w,y-head),(x+w+d_w,y+h),(153,153,0), 5)
       
        cm1[cm > 0] = 255
        cm1 = cv.erode(cm1,(5,5))
        cm1 = cv.dilate(cm1,(5,5))
    print(r)
        
    return cm1,r
def get_threshold(img,dep):
    index1,index2 = np.where(img>0)

    index1 = index1/1.5
    index2 = index2/1.5

    index1 = index1.astype('int64')
    index2 = index2.astype('int64')

    X = np.zeros(len(index1))
    for i in range(len(index1)):
        X[i] = dep[index1[i],index2[i]]
    X = X[np.nonzero(X)]
    X = X.reshape(-1,1)
    if( X.size == 0):
        return 215
    km = KMeans(n_clusters=2)

    km.fit(X)

    labels = km.labels_
    cluster_centers = km.cluster_centers_

    n_clusters_ = len(np.unique(labels))
    #print n_clusters_
    min_ = 65535
    for i in range(n_clusters_):
        print(cluster_centers[i])
        if(cluster_centers[i] < min_):
            min_ = cluster_centers[i]
            r_label = i
    
    my_member_ind = labels == r_label
    my_members  = X[my_member_ind]
    min_ = my_members.max()

    return min_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet101_voc2012',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default='./zmq2_4',
                        help='Path the input image')
    parser.add_argument('-th','--thres',type=int,default=2400)
    parser.add_argument('-s','--start',type=int,default = 0)
    parser.add_argument('-e','--end',type = int,default = 2000)
    parser.add_argument('-o', '--output_path', type=str, default='./zmq2_4/nxp',
                        help='Path to output rgb')
    parser.add_argument('--id', default="1")
    parser.add_argument('--input_size', type=int, default=500)
    parser.add_argument('-f', '--flip', type=bool, default=True,
                        help="Whether the network should predict on both image and flipped image.")

    args = parser.parse_args()
    image_path = args.input_path
    save_path = args.output_path 
    start_frame = args.start
    end_frame = args.end
    # Handle input and output args
    #images = glob(args.glob_path) if args.glob_path else [args.input_path,]
    #dep_images = glob(args.glob_depth_path) if args.glob_depth_path else[args.input_depth_path,] 
    
    #images = zip(images,dep_images)
    # if args.glob_path:
    #     fn, ext = splitext(args.output_path)
    #     if ext:
    #         parser.error("output_path should be a folder for multiple file input")
    #     if not isdir(args.output_path):
    #         os.mkdir(args.output_path)

    # Predict
    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
        if not args.weights:
            if "pspnet50" in args.model:
                pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                                  weights=args.model)
            elif "pspnet101" in args.model:
                if "cityscapes" in args.model:
                    pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                       weights=args.model)
                if "voc2012" in args.model:
                    pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                       weights=args.model)

            else:
                print("Network architecture not implemented.")
        else:
            pspnet = PSPNet50(nb_classes=2, input_shape=(
                768, 480), weights=args.weights)

        for i in range(start_frame,end_frame):
            print("Processing image {} / {}".format(i,args.end))
            img = misc.imread(image_path + "/rgb/fid_"+ str(i) + ".png" , mode='RGB')
            depth_img = cv.imread(image_path + "/dep/fid_" + str(i) + ".png",-1)
            
            cimg = misc.imresize(img, (args.input_size, args.input_size))

            #mask = using_depth_refined(depth_img)
            #img = cv.add(img,np.zeros(np.shape(img),dtype=np.uint8),mask=mask)

            # if args.glob_path:
            #     input_filename, ext = splitext(basename(img_path[0]))
            #     print(input_filename)
            #     if glob("./zmq1/nxp/"+input_filename+"_dep.png"):
            #         print("pass")
            #         continue
            #     filename = join(args.output_path, input_filename)
            #     #filename1 = join(args.output_path1,input_filename)
            # else:
            #     filename, ext = splitext(args.output_path)
            dep_img = depth_img.astype(np.uint8)
            threshold = args.thres
            dep_img[depth_img > threshold] = 0
            dep_img[depth_img <= threshold] = 255
            dep_img[:,0:50] = 0
            dep_img = cv.resize(dep_img,(768,636))
            
            #dep_img = dep_img.astype(np.uint8)
            img = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=dep_img)
            #cv.imwrite("img" + str(i) + ".png",img)
            
            probs = pspnet.predict(img, args.flip)

            cm = np.argmax(probs, axis=2)
            pm = np.max(probs, axis=2)

            color_cm = utils.add_color(cm)
            # color cm is [0.0-1.0] img is [0-255]
            #alpha_blended = 0.5 * color_cm * 255 + 0.5 * img

            
                #filename1,ext1 = splitext(args.output_path1)

            #img = cv.imread("fid_1.png",0)
            
            cm[cm != 15] = 0
            cm[cm == 15] = 1
            # cm = cm.astype(np.uint16)
            
            # thresh = get_threshold(cm,depth_img)
            # print(thresh)
            # mask = using_depth_refined(depth_img,thresh)
            
            
            # cm = np.multiply(cm,mask)
            
            #cm1,r = test_res(cm,mask)
            cm1 = cm.astype(np.uint8)
            cm1[cm > 0] = 255
            cm1 = cv.medianBlur(cm1,5)
            
            cm_dep = cv.resize(cm1,(512,424))
            nxp_img = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=cm1)
            nxp_dep = cv.add(depth_img,np.zeros(np.shape(depth_img),dtype=np.uint16),mask=cm_dep)
            #cv.waitKey(0)
            nxp_img = cv.cvtColor(nxp_img,cv.COLOR_RGB2BGR)
            
            filename = save_path + "/fid_" + str(i)
            

            cv.imwrite(filename+"_res"+".png",nxp_img)
            #filename1 = "./nxp/dep"
            cv.imwrite(filename+"_dep"+".png",nxp_dep)
            #cv.imwrite(filename+"mask"+ext,cm1)
            #cv.imwrite(filename+"clolor"+ext,color_cm)
            #scipy.io.savemat(filename + 'probs.mat', {'probs': pm, 'labels': cm})
