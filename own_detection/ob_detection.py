import os

import cv2

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class ObjectDetection():
    def __init__(self,model_name,model_folder,path_labels,num_classes):
        self.model_name = model_name
        self.model_folder = model_folder
        self.path_label = path_labels
        self.num_classes = num_classes
        self.path_model = os.path.join(model_folder,model_name)

        self._load_map()
        self._load_model()



    def _load_model(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)

        self.detection_image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.detection_num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def _load_map(self):
        self.label_map = label_map_util.load_labelmap(self.path_label)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes,                                                                 use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def detect_objects(self,image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        sess = self.sess
        # detection_graph = self.detection_graph
        #
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        #
        # # Each box represents a part of the image where a particular object was detected.
        # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        #
        # # Each score represent how level of confidence for each of the objects.
        # # Score is shown on the result image, together with the class label.
        # scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # num_detections = detection_graph.get_tensor_by_name('num_detections:0')



        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={self.detection_image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        # Actual detection.
        # (self.detection_boxes, self.detection_scores, self.detection_classes, self.detection_num_detections) = sess.run(
        #     [self.detection_boxes, self.detection_scores, self.detection_classes, self.detection_num_detections],
        #     feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.


        # plt.figure(figsize=IMAGE_SIZE)
        # plt.imshow(image_np)
        # plt.show()
        return image_np

def main():
    img = cv2.imread('test2.jpg')
    model_folder = '../ssd_mobilenet_v1_coco_11_06_2017'
    PATH_TO_LABELS = os.path.join('../object_detection/data', 'mscoco_label_map.pbtxt')

    object = ObjectDetection(model_name='frozen_inference_graph.pb',model_folder='../ssd_mobilenet_v1_coco_11_06_2017',path_labels=PATH_TO_LABELS,num_classes=20)

    for i in xrange(5):
        object.detect_objects(img)

if __name__ =='__main__':
    main()