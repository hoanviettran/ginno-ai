import argparse
import sys
import time
import cv2
import face
import datetime
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
    # detector = dlib.get_frontal_face_detector()

    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FOURCC, codec)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    # face_recognition = face.Recognition()
    face_detection = face.Detection()
    DetectEnable = 1

    while True:
    # Capture frame-by-frame
    # mypath = '/home/hoanviettran/Pictures/Face/TranTuanHung/facebook/TranTuanHung/'
    # files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for file in files:
    #     file_path = mypath + file
    #     print(file)
    #     frame = cv2.imread(file_path)
    #     frame_count += 1
        ret, frame = video_capture.read()
        if(ret):
            if(DetectEnable):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # bbs = detector(frame_rgb, 0)
                # display = detect.bb_face(frame_rgb,bbs)
                # faces = face_recognition.identify(frame_rgb, threshold)
                faces, images = face_detection.find_faces(frame_rgb)
                if(len(faces) > 0):
                    DetectEnable = 0
                    start = time.time()
            # Check our current fps


            # add_overlays(frame, faces, frame_rate)

            # cv2.imshow('Video', frame)
            if(DetectEnable == 0):
                cv2.imwrite('/mnt/data/Face/Frames/' + str(datetime.now()) + '.jpg', frame)

            end = time.time()
            if((end - start) > 5):
                DetectEnable = 1

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