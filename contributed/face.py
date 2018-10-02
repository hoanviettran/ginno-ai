# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
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

import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet
import time


gpu_memory_fraction = 0.3
facenet_model_checkpoint = "/mnt/data/pretrained-models/20180408-102900"
dict_path = "/mnt/data/facenet/contributed/Gino_dict_2018.pickle"
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        # if len(faces) == 1:
        face = faces[0]
        face.name = person_name
        face.embedding = self.encoder.generate_embedding(face)
        return faces

    def identify(self, image, threshold):
        start_find = time.time()
        faces, images = self.detect.find_faces(image)
        end_find = time.time()
        print('find_time', end_find-start_find)
        # for i, face in enumerate(faces):
        #     if debug:
        #         cv2.imshow("Face: " + str(i), face.image)
        #     face.embedding = self.encoder.generate_embedding(face)
        #     distance, name = self.identifier.identify(face)
        #     if(distance < threshold):
        #         face.name = name
        if(len(faces) > 0):
            start_enc = time.time()
            embeddings = self.encoder.generate_embeddings(images)
            end_enc = time.time()
            # print('enc_time', end_enc - start_enc)
            start_match = time.time()
            for i, face in enumerate(faces):
                distance, name = self.identifier.identify(embeddings[i])
                if (distance < threshold):
                    face.name = name
            end_match = time.time()
            print('enc-match-time', end_match - start_enc)
        return faces


class Identifier:
    def __init__(self):
        self.std_dict = {}
        with open(dict_path, 'rb') as dict:
            self.std_dict = pickle.load(dict)

    def load_dict(self, dict_path):
        with open(dict_path, 'rb') as dict:
            self.std_dict = pickle.load(dict)

    def identify(self, embedding):
        if embedding is not None:

            distances = np.linalg.norm(embedding - self.std_dict['embeddings_array'], axis=1)
            label_indx = np.argmin(distances)
            match_class = self.std_dict['name_array'][self.std_dict['labels_array'][label_indx]]
            distance = distances[label_indx]
            return distance, match_class


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def generate_embeddings(self, images):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)



class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self.setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def setup_mtcnn(self):
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session()
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []
        # start = time.time()

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        # end = time.time()
        # print('detect_time', end - start)
        # start = time.time()


        # images = []
        # for bb in bounding_boxes:
        #     face = Face()
        #     face.container_image = image
        #     face.bounding_box = np.zeros(4, dtype=np.int32)
        #     face_size = bb.bottom() - bb.top()
        #     margin_h = int(face_size * 0.1375)
        #     margin_w = int(face_size * 0.05)
        #     img_size = np.asarray(image.shape)[0:2]
        #     face.bounding_box[0] = np.maximum(bb.left() - margin_w, 0)
        #     face.bounding_box[1] = np.maximum(bb.top() - margin_h, 0)
        #     face.bounding_box[2] = np.minimum(bb.right() + margin_w, img_size[1])
        #     face.bounding_box[3] = np.minimum(bb.bottom() + margin_h, img_size[0])
        #     cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #     face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
        #     face.image = facenet.prewhiten(face.image)
        #     faces.append(face)
        #     images.append(face.image)
        # images_ar = np.array(images)

        bounding_boxes = bounding_boxes[np.argwhere(bounding_boxes[:, 4] > 0.95)[:, 0]]
        faces_sorted = np.argsort(
            (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1]))[::-1]

        bounding_boxes = bounding_boxes[faces_sorted]
        images = []
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            face.image = facenet.prewhiten(face.image)
            faces.append(face)
            images.append(face.image)
        images_ar = np.array(images)

        # end = time.time()
        # print('bb_time', end - start)
        return faces, images_ar

    def bb_face(self,image,dlib_bbs):
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        bounding_boxes = bounding_boxes[np.argwhere(bounding_boxes[:, 4] > 0.95)[:, 0]]
        for bb in bounding_boxes:
            cv2.rectangle(image, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])), (0,255,0), 2)
        for bb in dlib_bbs:
            cv2.rectangle(image, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0, 0, 255), 2)
        return image
