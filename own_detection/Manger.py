from threading import Thread
import cv2
import argparse
import sys
from multiprocessing import Queue, Process
import logging.config
from logging_config import LOGGING

from ob_detection import ObjectDetection
import time

logging.config.dictConfig(LOGGING)
log = logging.getLogger(__name__)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    #setting for video property
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')

    parser.add_argument('--width', type=int,
                        help='Width of the frames in video stream.', default=480)

    parser.add_argument('--height', type=int,
                        help='Height of the frames in video stream.', default=360)

    #setting for Queue
    parser.add_argument('--qsize',type = int,help='size of queue',default=30)

    parser.add_argument('--mode',type=str,help='debug(1) or release',default='debug')


    #setting for detection
    parser.add_argument('--model_folder', type=str, help='path where put the pre-trained model', default='../ssd_mobilenet_v1_coco_11_06_2017')
    #parser.add_argument('--model_folder', type=str, help='path where put the pre-trained model',default='/home/tcl-admin/Downloads/ssd_inception_v2_coco_11_06_2017')
    parser.add_argument('--model_name', type=str, help='pre-trained model name',
                        default='frozen_inference_graph.pb')

    parser.add_argument('--label_mapping', type=str, help=' indices to category names',
                        default='../object_detection/data/mscoco_label_map.pbtxt')
    return parser.parse_args(argv)

class Manager():

    def __init__(self,video_srouce,qsize,mode,width,height):
        video_srouce = "/home/tcl-admin/Downloads/pets2006_1.avi"
        self.video_capture = cv2.VideoCapture(video_srouce)
        self.video_capture.set(3, width)
        self.video_capture.set(4, height)

        #may group togather as one object
        self.input_queue = Queue(qsize)
        self.output_queue = Queue(qsize)

        self.mode = 1 if mode==0 else 0

    def read_input_frame(self):
        return self.input_queue.get()

    def read_output_frame(self):
        return self.output_queue.get()

    def proecess_thread(self,args):
        t1 = Process(name='detection process', target=self.child, args=(args,))
        t1.start()


    def child(self,args):
        object_detector = ObjectDetection(model_name=args.model_name, num_classes=90, path_labels=args.label_mapping,
                                      model_folder=args.model_folder)
        while True:
            start_time = time.time()
            result = object_detector.detect_objects(self.input_queue.get())
            print('Detection Process time %.3f' % (time.time() - start_time))
            self.output_queue.put(result)



def main(args):


    manger = Manager(video_srouce=args.video_source,qsize=args.qsize,width=args.width,height=args.height,mode=args.mode)
    manger.proecess_thread(args)

    while True:
        start_time = time.time()
        _, frame = manger.video_capture.read()
        manger.input_queue.put(frame)
        print('Read Frame time %.3f' % (time.time() - start_time))

        # manger.output_queue.put(result)
        cv2.imshow('Video', manger.read_output_frame())
        cv2.waitKey(1)






    pass

if __name__ =='__main__':
    main(parse_arguments(sys.argv[1:]))