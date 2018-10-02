# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
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
import argparse
import sys
import time
import cv2
import face
from os import listdir
from os.path import isfile, join
import dlib

def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main(args):
    frame_interval = 2  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    threshold = 0.77
    detector = dlib.get_frontal_face_detector()

    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FOURCC, codec)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    face_recognition = face.Recognition()

    start_time = time.time()
    cap_time = 0
    iden_time = 0
    addt_time = 0
    # if args.debug:
    #     print("Debug enabled")
    #     face.debug = True
    # detect = face.Detection()
    while True:
    # Capture frame-by-frame
    # mypath = '/home/hoanviettran/Pictures/Face/TranTuanHung/facebook/TranTuanHung/'
    # files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for file in files:
    #     file_path = mypath + file
    #     print(file)
    #     frame = cv2.imread(file_path)
        frame_count += 1
        # start_cap = time.time()
        ret, frame = video_capture.read()
        # if (frame_count % frame_interval) == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # end_cap = time.time()
        # cap_time += end_cap - start_cap
        start_iden = time.time()
        # bbs = detector(frame_rgb, 0)


        # display = detect.bb_face(frame_rgb,bbs)
        faces = face_recognition.identify(frame_rgb, threshold)

        # Check our current fps
        end_time = time.time()
        iden_time += end_time - start_iden
        if (end_time - start_time) > fps_display_interval:
            frame_rate = int(frame_count / (end_time - start_time))
            start_time = time.time()
            print('---')
            print('cap_time:', cap_time/frame_count)
            print('iden_time:', iden_time/frame_count)
            frame_count = 0

        start_addt = time.time()
        add_overlays(frame, faces, frame_rate)
        end_addt = time.time()
        addt_time += end_addt - start_addt

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
