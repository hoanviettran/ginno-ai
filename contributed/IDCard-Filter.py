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
from tqdm import tqdm
import cv2
import face
from os import listdir
from os.path import isfile, join, isdir
import os
import pandas as pd

def main(args):

    df_profile = pd.read_excel('/mnt/data/cmnd_tima/blacklist.xlsx', sheet_name='Profile')
    df_image = pd.read_excel('/mnt/data/cmnd_tima/blacklist.xlsx', sheet_name='Image')
    face_detect = face.Detection()
    personIDs = [f for f in listdir(args.rawpath) if isdir(join(args.rawpath, f))]
    for personID in tqdm(personIDs):
        cmnd = df_profile[df_profile['CustomerCreditId'] == int(personID)]['CardNumber'].values[0]
        i = 0
        personpath = args.rawpath + personID
        #filteredpath = args.pscard_dir + personID
        #if os.path.isdir(filteredpath):
            #continue
        #os.system('mkdir ' + filteredpath)
        files = [f for f in listdir(personpath) if isfile(join(personpath, f))]
        for file in files:
            try:
                filepath = personpath + '/' + file
                img = cv2.imread(filepath)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces, images = face_detect.find_faces(img_rgb)
                if(len(faces) > 0):
                    i += 1
                    os.system('cp ' + filepath + ' ' + args.pscard_dir + cmnd + '_' + str(i) + '.' + file.split('.')[-1])
            except:
                pass

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--rawpath', type=str,
                        help='Rawcards directory', default='/home/hoanviettran/tima-blacklist/')
    parser.add_argument('--pscard_dir', type= str,
                        help='Person card directory', default='/home/hoanviettran/tima-blacklist-face/')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
