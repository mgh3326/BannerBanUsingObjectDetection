# Import packages
import json
import math
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import csv
import datetime
import folium
import base64
from folium import IFrame

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


# searchName = "바나나 알러지 원숭이"


#

def is_true_banner(location):
    f = open('output.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    # my = [37.2523317, 127.219356]
    oh_min = 1.0774115445377469e-07
    # oh_min = 5.7924941194268736e-08
    distance = []
    for line in rdr:
        if line[1] == "latitude":
            continue
        distance.append(math.pow(location[0] - float(line[1]), 2) + math.pow(location[1] - float(line[2]), 2))
    # print(distance)
    print(min(distance))
    f.close()
    return min(distance) <= oh_min


# m_net(input())
class BannerDetection:
    mArtist = ""
    mAlbum = ""
    mYear = ""
    # mTrack = ""
    mGenre = ""
    mComment = ""
    mImagePath = ""
    mLyric = ""
    mOutput = ""

    # 초기자(initializer)

    def __init__(self):
        # self.* : 인스턴스변수

        self.create()

    # 메서드
    def create(self):
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph'
        # IMAGE_NAME = 'test2.jpg'

        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

        # Path to image

        # Number of classes the object detector can identify
        NUM_CLASSES = 5

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value

    def detection(self, _image, _min_score=0.8, _name="temp", lat=0, loc=0):
        image = cv2.imread(_image)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=_min_score)
        outstr = ""
        post_percent = 0
        post = [0, 0, 0, 0]
        count = 0
        for i in range(scores.size):
            count = count + 1
            if scores[0][i] < _min_score:
                break
            outstr += self.category_index[classes[0][i]]['name'] + "(" + str(
                scores[0][i]) + ")\n"
            print(self.category_index[classes[0][i]]['name'] + "(" + str(
                scores[0][i]) + ")\n")
            if self.category_index[classes[0][i]]['name'] == "post":
                if scores[0][i] > post_percent:
                    post_percent = scores[0][i]
                    post = [boxes[0][i][1], boxes[0][i][0], boxes[0][i][3], boxes[0][i][2]]
        # if post_percent > 0 and is_true_banner([lat, loc]):
        if post_percent > 0:

            outstr += "합법 현수막이 포착되었습니다.\n합법 현수막안의 배너는 불법 현수막에서 예외처리가 됩니다.\n"
            for i in range(count - 1):

                if self.category_index[classes[0][i]]['name'] == "banner":
                    if post[0] < boxes[0][i][1] and post[1] < boxes[0][i][0] and post[2] > boxes[0][i][3] and post[3] > \
                            boxes[0][i][2]:  # 내부의 있을경우
                        continue
                    outstr += "불법 현수막이 포착되었습니다!!!\n"
                    image = cv2.rectangle(image, (int(image.shape[1] * boxes[0][i][1])
                                                  , int(image.shape[0] * boxes[0][i][0])),
                                          (int(image.shape[1] * boxes[0][i][3])
                                           , int(image.shape[0] * boxes[0][i][2])), (0, 0, 255), 3)
                    cv2.putText(image, '            BannerBan', (int(image.shape[1] * boxes[0][i][1])
                                                                 , int(image.shape[0] * boxes[0][i][0])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2)
        else:
            for i in range(count - 1):
                if self.category_index[classes[0][i]]['name'] == "banner":
                    outstr += "불법 현수막이 포착되었습니다!!!\n"

                    image = cv2.rectangle(image, (int(image.shape[1] * boxes[0][i][1])
                                                  , int(image.shape[0] * boxes[0][i][0])),
                                          (int(image.shape[1] * boxes[0][i][3])
                                           , int(image.shape[0] * boxes[0][i][2])), (0, 0, 255), 3)
                    cv2.putText(image, '          BannerBan', (int(image.shape[1] * boxes[0][i][1])
                                                               , int(image.shape[0] * boxes[0][i][0])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2)
        _time = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        cv2.imwrite('./out/' + _time + '.jpg', image)
        cv2.imwrite(_name + '.jpg', image)
        with open('picture_data.json', 'r', encoding='utf-8') as f:  # 가입이 되어있는지 확인
            try:
                datas = json.load(f)
            except json.decoder.JSONDecodeError:
                datas = list()
                member = {"mLatitude": lat, "mLongitude": loc, "mImagePath": './out/' + _time + '.jpg'}
                datas.append(member)
            else:
                member = {"mLatitude": lat, "mLongitude": loc, "mImagePath": './out/' + _time + '.jpg'}
                datas = list(datas)
                datas.append(member)
            with open('picture_data.json', 'w', encoding="utf-8") as outfile:
                json.dump(datas, outfile, ensure_ascii=False)
        mapa = folium.Map(location=[lat, loc], zoom_start=17)
        with open('picture_data.json', 'r', encoding='utf-8') as f:  # 위도 경도를 넣어주자
            datas = json.load(f)
            for data in datas:
                encoded = base64.b64encode(open(data["mImagePath"], 'rb').read()).decode()
                html = '<img src="data:image/jpeg;base64,{}"style="max-width:100%;">'.format
                iframe = IFrame(html(encoded), width=800 + 20, height=590 + 20)
                popup = folium.Popup(iframe, max_width=2650)

                icon = folium.Icon(color="red", icon="ok")
                marker = folium.Marker(location=[data["mLatitude"], data["mLongitude"]], popup=popup, icon=icon)
                mapa.add_child(marker)

        mapa.save('./out/index.html')

        return outstr
