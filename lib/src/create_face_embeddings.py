# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
from os import listdir
from os.path import isfile, join, isdir
import sys
from scipy import misc
import math
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import numpy as np
import cv2
from align import detect_face
import retrieve
import pickle

def main(args):

    model_exp = '/mnt/data/Face_Recognition-vinayak/lib/src/ckpt/20180402-114759'
    graph_fr = tf.Graph()
    sess_fr = tf.Session(graph=graph_fr)

    with graph_fr.as_default():
        saverf = tf.train.import_meta_graph(os.path.join(model_exp, 'model-20180402-114759.meta'))
        saverf.restore(sess_fr, os.path.join(model_exp, 'model-20180402-114759.ckpt-275'))
        pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)


    with tf.Graph().as_default():

        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = args.image_size
            # embedding_size = embeddings.get_shape()[1]
            extracted_dict = {}

            # Run forward pass to calculate embeddings
            src_path = '/media/hoanviettran/Seagate Backup Plus Drive/FaceDataset/train_celebrity/celebrity/'
            person_dirs = [f for f in listdir(src_path) if isdir(join(src_path, f))]
            # for person_dir in person_dirs:
            name = 0
            for person_dir in tqdm(person_dirs):
                # print(person_dir + '-' + str(i))
                # for filename in os.listfile(person_path):
                person_path = src_path + person_dir + '/'
                file_names = [person_path + f for f in listdir(person_path)]
                # file_names = [f for f in listdir(src_path) if (f[-3:] == 'tif')]
                # nrof_samples = int(len(file_names)/3) + 1
                nrof_samples = 10
                #nrof_samples = len(file_names)
                images_face = np.zeros((nrof_samples, image_size, image_size, 3))
                for j, file_name in enumerate(file_names):
                    img = cv2.imread(file_name)
                    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    response, faces, bboxs = retrieve.align_face(img_rgb, pnet, rnet, onet)
                    #crop = sorted(faces, key=np.size)[0]
                    #cv2.imwrite('/mnt/data/Face_dataset/Gino/' + str(name) + '.jpg',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                    #name += 1
                    images = facenet.load_img(faces[0], image_size)
                    images_face[j,:,:,:] = images[0,:,:,:]
                    if (j == nrof_samples - 1):
                        break

                feed_dict = {images_placeholder:images_face, phase_train_placeholder:False}
                feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                extracted_dict[person_dir] = feature_vector

            print("completed encoding images")
            with open('Asian_dict_2018.pickle', 'wb') as f:
                pickle.dump(extracted_dict, f)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--model', type=str,default='/mnt/data/Face_Recognition-vinayak/lib/src/ckpt/20180402-114759/',
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
