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
    #large than 200, smaller than 50 = 0
    ret,thresh1 = cv.threshold(img,215,255,cv.THRESH_TOZERO_INV)
    ret2,thresh2 = cv.threshold(thresh1,50,255,cv.THRESH_BINARY)
    thresh3 = cv.resize(thresh2,(768,636))
    dilated_res = cv.dilate(thresh3,(5,5))

    return dilated_res

def test_res(cm,mask):
    cm = cm.astype(np.uint8)
    contours, hierarchy = cv.findContours(cm,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    index = 0
    if len(contours) == 0:
        return cm,0

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
    if(r > 0.03):
        head = int(h/5)
        h = int(h-h/3)
        d_w = int(0.2*w)
        if(x-d_w <= 0):
            d_w = x
        if(x+w+d_w >= cm.shape[1]):
            d_w = cm.shape[1] - 1 - x -w
        cm[y-head:y+h,x-d_w:x+w+d_w] = mask[y-head:y+h,x-d_w:x+w+d_w] + cm[y-head:y+h,x-d_w:x+w+d_w]
        cm[cm > 0] = 255
        cm = cv.erode(cm,(5,5))
        cm = cv.dilate(cm,(5,5))
        
    return cm,r
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet101_voc2012',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default='./1_rgb/fid_95.png',
                        help='Path the input image')
    parser.add_argument('-id','--input_depth_path',type=str,default='./1_depth/fid_95.png')
    parser.add_argument('-g', '--glob_path', type=str, default='./1_rgb/*.png',
                        help='Glob path for multiple images')
    parser.add_argument('-gd','--glob_depth_path',type=str,default='./1_depth/*.png')
    parser.add_argument('-o', '--output_path', type=str, default='./nxp/rgb',
                        help='Path to output rgb')
    parser.add_argument('-o1', '--output_path1', type=str, default='./nxp/dep',
                        help='Path to output depth')
    parser.add_argument('--id', default="0")
    parser.add_argument('--input_size', type=int, default=500)
    parser.add_argument('-f', '--flip', type=bool, default=True,
                        help="Whether the network should predict on both image and flipped image.")

    args = parser.parse_args()

    # Handle input and output args
    images = glob(args.glob_path) if args.glob_path else [args.input_path,]
    dep_images = glob(args.glob_depth_path) if args.glob_depth_path else[args.input_depth_path,] 
    
    images = zip(images,dep_images)
    if args.glob_path:
        fn, ext = splitext(args.output_path)
        if ext:
            parser.error("output_path should be a folder for multiple file input")
        if not isdir(args.output_path):
            os.mkdir(args.output_path)

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

        for i, img_path in enumerate(images):
            print("Processing image {} / {}".format(i+1,len(images)))
            img = misc.imread(img_path[0], mode='RGB')
            depth_img = cv.imread(img_path[1],0)
            cimg = misc.imresize(img, (args.input_size, args.input_size))

            mask = using_depth_refined(depth_img)
            img = cv.add(img,np.zeros(np.shape(img),dtype=np.uint8),mask=mask)

            probs = pspnet.predict(img, args.flip)

            cm = np.argmax(probs, axis=2)
            pm = np.max(probs, axis=2)

            color_cm = utils.add_color(cm)
            # color cm is [0.0-1.0] img is [0-255]
            alpha_blended = 0.5 * color_cm * 255 + 0.5 * img

            if args.glob_path:
                input_filename, ext = splitext(basename(img_path[0]))
                filename = join(args.output_path, input_filename)
                filename1 = join(args.output_path1,input_filename)
            else:
                filename, ext = splitext(args.output_path)

            #img = cv.imread("fid_1.png",0)
            
            cm[cm != 15] = 0
            cm[cm == 15] = 255
            cm = cm.astype(np.uint8)
            #cm = np.multiply(cm,mask)
            cm,r = test_res(cm,mask)
            cm = cv.medianBlur(cm,5)
            cm_dep = cv.resize(cm,(512,424))
            nxp_img = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=cm)
            nxp_dep = cv.add(depth_img,np.zeros(np.shape(depth_img),dtype=np.uint8),mask=cm_dep)
            #cv.imshow("cm",cm)
            #cv.waitKey(0)      
            #misc.imsave(filename + "_seg_read" + ext, cm)
            #misc.imsave(filename + "_seg" + ext, color_cm)
            #misc.imsave(filename + "_probs" + ext, pm)
            #misc.imsave(filename + "_seg_blended" + ext, alpha_blended)
            print(filename)
            cv.imwrite(filename+"_res"+ext,nxp_img)
            #filename1 = "./nxp/dep"
            cv.imwrite(filename1+"_res"+ext,nxp_dep)
            #scipy.io.savemat(filename + 'probs.mat', {'probs': pm, 'labels': cm})