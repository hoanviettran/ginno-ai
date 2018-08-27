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
from facenet import load_data, load_img, load_model, to_rgb
#import lfw
import os
import sys
import math
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import cv2
from os import listdir
from os.path import isfile, join, isdir

from align import detect_face
from datetime import datetime
from scipy import ndimage
from scipy.misc import imsave
from scipy.spatial.distance import cosine
import pickle
import matplotlib.pyplot as plt
from numba import jit, float32, guvectorize, void, vectorize, njit, prange
#   face_cascade = cv2.CascadeClassifier('out/face/haarcascade_frontalface_default.xml')
parser = argparse.ArgumentParser()


parser.add_argument('--lfw_batch_size', type=int,
                    help='Number of images to process in a batch in the LFW test set.', default=100)
parser.add_argument('--image_size', type=int,
                    help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--detect_multiple_faces', type=bool,
                    help='Detect and align multiple faces per image.', default=True)
parser.add_argument('--margin', type=int,
                    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
parser.add_argument('--random_order',
                    help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
parser.add_argument('--gpu_memory_fraction', type=float,
                    help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

args = parser.parse_args()


def align_face(img, pnet, rnet, onet):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # print("before img.size == 0")
    if img.size == 0:
        print("empty array")
        return False, img, [0, 0, 0, 0]

    if img.ndim < 2:
        print('Unable to align')

    if img.ndim == 2:
        img = to_rgb(img)

    img = img[:, :, 0:3]

    bounding_boxes, _ = detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor)
    bounding_boxes = bounding_boxes[np.argwhere(bounding_boxes[:,4] > 0.95)[:,0]]
    faces_sorted = np.argsort((bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1]))[::-1]
    nrof_faces = bounding_boxes.shape[0]

    if nrof_faces == 0:
        return False, img, [0, 0, 0, 0]
    else:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if args.detect_multiple_faces:
                for i in faces_sorted:
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0]+det[:, 2])/2-img_center[1], (det[:, 1]+det[:, 3])/2-img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                # some extra weight on the centering
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))
        # if len(det_arr) > 0:
        faces = []
        bboxes = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-args.margin/2, 0)
            bb[1] = np.maximum(det[1]-args.margin/2, 0)
            bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(
                cropped, (args.image_size, args.image_size), interp='bilinear')
            # misc.imsave("cropped.png", scaled)
            faces.append(scaled)
            bboxes.append(bb)
            # print("leaving align face")
        return True, faces, bboxes

@jit(['float32(float32[:],float32[:])'])
# @jit()
def compare_feature(vector, array):
    return np.linalg.norm(vector - array)

# @jit()
def identify_person(image_vector, feature_array):
    distance_min = 2
    result = ''
    for j, person_features in enumerate(feature_array.values()):
        num_features = person_features.shape[0]
        for i in range (0,num_features):
            distance_ps = compare_feature(image_vector[0,:], person_features[i,:])
            if(distance_ps < distance_min):
                distance_min = distance_ps
                    # person_id = j
                result = list(feature_array.keys())[j]
    #     distance_ps = compare_feature(image_vector, person_features)
    #     if(distance_ps < distance_min):
    #         distance_min = distance_ps
    #         person_id = j
    # match = distance_min < threshold
    # result = list(feature_array.keys())[person_id]
    return distance_min, result

def detectface_camera(sess, pnet, rnet, onet, feature_array):
    # Get input and output tensors
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder, (160, 160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
    image_size = args.image_size

    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    name = 30
    while(True):

        ret, frame = cap.read()
        if (ret):
        # frame =             .imread('/mnt/data/Face_dataset/Gino/fr41.png')
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            response, faces, bboxs = align_face(frame_rgb, pnet, rnet, onet)
            frame_show = frame.copy()
            if (response == True):
                for i, face_crop in enumerate(faces):
                    bb = bboxs[i]
                    images = load_img(face_crop, image_size)
                    feed_dict = {images_placeholder: images,
                                 phase_train_placeholder: False}
                    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                    distance_min, result = identify_person(feature_vector, feature_array)
                    if (distance_min < 0.77):
                        W = int(bb[2] - bb[0]) // 2
                        # H = int(bb[3] - bb[1]) // 2
                        cv2.putText(frame_show, result, (bb[0] + W - (
                                W//2), bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame_show, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

            cv2.imshow('img', frame_show)
            key =  cv2.waitKey(1)
            if(key == ord('q')):
                cap.release()
                cv2.destroyAllWindows()
                break
        # elif(key == ord('n')):
        #     face_sample = cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR)
        #     cv2.imshow('Sample', face_sample)
        #     key = cv2.waitKey()
        #     if (key == ord('y')):
        #         person_name = input('Enter your name: ')
        #         feed_dict = {images_placeholder: face_sample, phase_train_placeholder: False}
        #         feature_vector = sess.run(embeddings, feed_dict=feed_dict)
        #         feature_array[person_name] = feature_vector
            elif(key == ord('c')):
                cv2.imwrite('/mnt/data/Face_dataset/Gino/' + str(name) + '.jpg',frame)
                name = name + 1
    #
    # print("encoding new person")
    # with open('extracted_dict.pickle', 'wb') as f:
    #     pickle.dump(feature_array, f)

def recognize_face(sess, pnet, rnet, onet, feature_array):

    # Get input and output tensors
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder, (160, 160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = args.image_size
    # embedding_size = embeddings.get_shape()[1]

    #     cap = cv2.VideoCapture(-1)
    #     while(True):
    test_path = '/mnt/data/Face_dataset/lfw/'
    thresholds = np.arange(1.0, 0.75, -0.01)
    cases = len(thresholds)
    num_faces = 0
    num_registedface = 0
    true_detect = np.zeros(cases)
    false_detect = np.zeros(cases)
    person_dirs = [f for f in listdir(test_path) if isdir(join(test_path, f))]
    extract = np.zeros(0)
    matching = np.zeros(0)
    monte = 0
    for k, person_dir in enumerate(person_dirs):
        print(person_dir + str(k))
        person_path = test_path + person_dir + '/'
        files = [f for f in listdir(person_path) if isfile(join(person_path, f))]
        # files = [f for f in listdir(person_path) if (f[-3:] == 'tif')]
        test_indx = int(len(files)/3)
        for file in files[test_indx:]:
            file_path = person_path + file
            # img_cv = cv2.imread(file_path)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = scipy.misc.imread(file_path)
            # # if cv2.waitKey(1) & 0xFF == ord('q'):
            # #         cap.release()
            # #         cv2.destroyAllWindows()
            # #         break
            # if (gray.size > 0):
            #     print(gray.size)
            #     # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            response, faces, bboxs = align_face(frame, pnet, rnet, onet)
            # #     print(response)
            if (response):
                for i, face_crop in enumerate(faces):

                    images = load_img(face_crop, image_size)
                    start_extract = time.time()
                    feed_dict = {images_placeholder: images,
                                 phase_train_placeholder: False}
                    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                    end_extract = time.time()
                    extract = np.append(extract, end_extract - start_extract)
                    start_matching = time.time()
                    distance_min, result = identify_person(feature_vector, feature_array)
                    end_matching = time.time()
                    matching = np.append(matching, end_matching - start_matching)
                    start_monte = time.time()
                    for j, threshold in enumerate(thresholds):
                        if(distance_min < threshold):
                            # print(str(distance_min) + result + file + str(i))
                            if(result == person_dir):
                                true_detect[j] = true_detect[j] + 1
                                # bb = bboxs[i]
                                # cv2.rectangle(img_cv, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                # W = int(bb[2] - bb[0]) // 2
                                # # H = int(bb[3] - bb[1]) // 2
                                # cv2.putText(img_cv, result, (bb[0] + W - (
                                #         W//2), bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                # print("TRUE-" + result)
                            else:
                                # print("FALSE" + result + file_path)
                                false_detect[j] = false_detect[j] + 1
                                # cv2.imshow('fail', face_crop)
                                # if(cv2.waitKey() == ord('q')):
                                #     cv2.destroyAllWindows()
                                #     break
                                # cv2.destroyAllWindows()
                        else:
                            break
                    end_monte = time.time()
                    monte = monte + end_monte - start_monte
                num_faces = num_faces + 1
            num_registedface = num_registedface + 1
            # cv2.imshow('detected', img_cv)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    # print("False acceptance rate: %f"%(false_detect/num_faces))
    # print("False rejection rate: %f"%((num_registedface-true_detect)/num_faces))
    print("average extract time: %f"%(extract.mean()))
    print("average matching time: %f" %(matching.mean()))
    print("average monte time: %f" %(monte))
    plt.figure()
    plt.plot(thresholds, false_detect/num_faces, label='False acceptance rate')
    plt.plot(thresholds, (num_registedface - true_detect)/num_faces, label='False rejection rate')
    plt.legend()
    plt.show()