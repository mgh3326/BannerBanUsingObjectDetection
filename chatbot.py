# Import packages
import json
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image

sys.path.append("..")
import telepot

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import scipy.misc
from time import sleep
from mypackage import BannerDetection
from mypackage import chat_member


def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id)

    with open('data.json', 'r', encoding='utf-8') as f:  # 가입이 되어있는지 확인
        try:
            datas = json.load(f)
        except json.decoder.JSONDecodeError:
            datas = list()
            member = chat_member.chat_memeber()
            member.mId = 0
            member.mLatitude = 0
            member.mLongitude = 0
            datas.append(member.__dict__)

        for data in datas:
            if data["mId"] == chat_id:
                break
            elif data["mId"] == datas[-1]["mId"]:
                member = chat_member.chat_memeber()
                member.mId = chat_id
                member.mLatitude = 0
                member.mLongitude = 0
                datas.append(member.__dict__)
                bot.sendMessage(
                    chat_id, text='처음 오셨군요 가입이 되었습니다.')
                with open('data.json', 'w', encoding="utf-8") as outfile:
                    json.dump(datas, outfile, ensure_ascii=False)

    if content_type == 'text':
        if msg['text'] == '/start':  # 시작
            bot.sendMessage(
                chat_id,
                text='위치정보를 보내주세요')
        else:

            for data in datas:
                if data["mId"] == chat_id:
                    bot.sendMessage(
                        chat_id,
                        text='현재 위도 경도 입니다. 사진을 보내주세요.\nlatitude : ' + str(data["mLatitude"]) + '\nlongitude : ' + str(
                            data["mLongitude"]))
    elif content_type == 'photo':
        bot.download_file(msg['photo'][-1]['file_id'], './file.png')
        # _image = Image.open("./file.png")
        bot.sendMessage(
            chat_id, text='사진을 전송이 완료되었습니다.')
        # PATH_TO_IMAGE = os.path.join(CWD_PATH,_image)
        with open('data.json', 'r', encoding='utf-8') as f:  # 위도 경도를 넣어주자
            datas = json.load(f)
            for data in datas:
                if data["mId"] == chat_id:
                    outstr = var.detection(_image="file.png", _min_score=0.8, _name=str(chat_id), lat=data["mLatitude"],
                                           loc=data["mLongitude"])

                    bot.sendMessage(
                        chat_id,
                        text='현재 제보자의 위도 경도 입니다.\nlatitude : ' + str(data["mLatitude"]) + '\nlongitude : ' + str(
                            data["mLongitude"]) + "\nmgh3326.iptime.org\n위 링크는 시각화 링크입니다.")
                    if outstr != "":
                        bot.sendMessage(
                            chat_id, text=outstr)
        bot.sendPhoto(chat_id, open(str(chat_id) + ".jpg", 'rb'))
    elif content_type == 'location':
        latitude = msg['location']['latitude']
        longitude = msg['location']['longitude']
        with open('data.json', 'r', encoding='utf-8') as f:  # 위도 경도를 넣어주자
            datas = json.load(f)
            for data in datas:
                if data["mId"] == chat_id:
                    data["mLatitude"] = latitude
                    data["mLongitude"] = longitude
        with open('data.json', 'w', encoding="utf-8") as outfile:
            json.dump(datas, outfile, ensure_ascii=False)
        bot.sendMessage(
            chat_id, text='위도 경도를 업데이트 하였습니다.\nlatitude : ' + str(latitude) + '\nlongitude : ' + str(longitude))
    elif content_type == 'video':
        bot.download_file(msg['video']['file_id'], './file.mp4')
        bot.sendMessage(
            chat_id, text='동영상 전송이 완료되었습니다.')
        video = cv2.VideoCapture('file.mp4')
        # out = cv2.VideoWriter('output3.avi', -1, 20.0, (640, 480))
        w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(str(chat_id) + '.mp4', fourcc, 15.0, (int(w), int(h)))

        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # out = cv2.VideoWriter('output3.mp4',fourcc, 20.0, (640, 480))
        while (video.isOpened()):

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = video.read()
            # out.write(frame)
            frame_expanded = np.expand_dims(frame, axis=0)
            # Perform the actual detection by running the model with the frame as input
            try:
                (boxes, scores, classes, num) = var.sess.run(
                    [var.detection_boxes, var.detection_scores, var.detection_classes, var.num_detections],
                    feed_dict={var.image_tensor: frame_expanded})
            except TypeError:
                break
            _min_score = 0.8
            outstr = ""

            # if post_percent > 0 and is_true_banner([lat, loc]):
            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                var.category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=_min_score)
            count = 0
            for i in range(scores.size):
                count = count + 1
                if scores[0][i] < _min_score:
                    break
                if var.category_index[classes[0][i]]['name'] == "banner":
                    bot.sendMessage(
                        chat_id, text='불법 현수막이 포착되었습니다.!!')
                    cv2.imwrite(str(chat_id) + str(count) + '.jpg', frame)
                    bot.sendPhoto(chat_id, open(str(chat_id) + str(count) + ".jpg", 'rb'))

            out.write(frame)
            # # All the results have been drawn on the frame, so it's time to display it.
            # cv2.imshow('Object detector', frame)
            #
            # # Press 'q' to quit
            # if cv2.waitKey(1) == ord('q'):
            #     break

        # Clean up
        video.release()
        out.release()
        cv2.destroyAllWindows()
        bot.sendVideo(chat_id, open(str(chat_id) + ".mp4", 'rb'))
    else:
        bot.sendMessage(
            chat_id, text='음악에 대해 자유롭게 검색을 해주세요')


if __name__ == "__main__":
    var = BannerDetection.BannerDetection()
    member = chat_member.chat_memeber()
    print("Start")
    with open('config/secret.json') as f:
            data = json.load(f)
    bot = telepot.Bot(data["telepot_key"])
    bot.message_loop(handle)
    print('Listening ...')

    while 1:
        # sleep(randint(10, 20))
        sleep(10)
