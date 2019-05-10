from PIL import Image, ImageDraw
import face_recognition
import cv2
import time
from os import path as osp
import os

def imresize(img, inp_dim=(416, 416)):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_image
# Load the jpg file into a numpy array
video_capture = cv2.VideoCapture('C:/Users/wty/Desktop/faceimage/正一.avi')

filepath = 'C:/Users/wty/Desktop/faceimage'
path = osp.join(filepath,'zy')
left_eye = osp.join(path,'left')
right_eye = osp.join(path,'right')
if not osp.exists(left_eye):
    os.makedirs(left_eye)
if not osp.exists(right_eye):
    os.makedirs(right_eye)

start = time.time()
i = 0
index = 0
while True:
    i+=1
    _,image = video_capture.read()
    # image=imresize(image)
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image,model='small')

    # print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    # Create a PIL imagedraw object so we can draw on the picture
    # pil_image = Image.fromarray(image)
    # d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        # for facial_feature in face_landmarks.keys():
        #     print("The {} in this face has the following points: {}".format(facial_feature,
        #                                                                     face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            # d.line(face_landmarks[facial_feature], width=5)
            local_list = face_landmarks[facial_feature]
            if facial_feature.endswith('eye'):
                local_list = face_landmarks[facial_feature]
                eye_local_1 = local_list[0]  # 眼睛左边点坐标
                eye_local_2 = local_list[1]  # 眼睛右边点坐标
                width = (eye_local_2[0] - eye_local_1[0])  # 右坐标列数-左坐标列数
                hight = int(width * (3 / 4))
                left_top = (int(eye_local_1[0]-0.6*width), eye_local_1[1] - int(hight*1.6 / 2))
                right_bottom = (int(eye_local_2[0]+0.6*width), eye_local_1[1] + int(hight*1.6 / 2))

                #裁剪左右眼
                # if facial_feature == 'right_eye':
                #     try:
                #         cut_image = image[right_bottom[1]:left_top[1],right_bottom[0]:left_top[0]]
                #         cv2.imshow('a', cut_image)
                #         # cv2.imwrite('1.jpg',cut_image)
                #         cv2.imwrite(osp.join(right_eye,'{}.jpg'.format(index)), cut_image)
                #     except Exception:
                #         continue
                # if facial_feature == 'left_eye':
                #     cv2.imwrite(osp.join(left_eye,'{}.jpg'.format(index)), cut_image)
                cv2.rectangle(image, left_top, right_bottom, (0, 0, 255), 1)

                index+=1
                # for local in local_list:
                    # cv2.circle(image,local,1,(255,255,255),4)

    if i%20==0:
        print("threading {} FPS of the video is {:5.4f}".format(0, i / (time.time() - start)))
    cv2.imshow('a',image)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()