import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import face_recognition
import cv2
import time
from os import path as osp
import os
from PIL import Image

# 读取训练数据
def loadtraindata():
    path = r"/home/software_mount/detection/dataset"                                         # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((24, 24)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.CenterCrop(24),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True)
    return trainloader

# 定义网络结构
class Net(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 12, 5)  # 卷积层
        self.fc1 = nn.Linear(12 * 3 * 3, 80)  # 全连接层
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 2)  # 2个输出



    def forward(self, x):  # 前向传播

        x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12*3*3)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        # 从卷基层到全连接层的维度转换

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
classes = ('0','1')
net = None
transformer = transforms.Compose([
                                                    transforms.Resize((24, 24)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])

# 加载测试集
def loadtestdata():
    path = r"/home/software_mount/detection/dataset"
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((24, 24)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=True)
    return testloader

# 训练并保存
def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分
    for epoch in range(10):  # 训练的数据量为10个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN网络net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.item()  # loss累加
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数

# 从磁盘读取模型及参数
def reload_net():
    print("尝试加载本地模型")
    trainednet = torch.load('net.pkl')
    if trainednet:
        print("加载本地模型成功")
    return trainednet
# 显示图片
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imresize(img, inp_dim=(416, 416)):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_image

def predict(image):
    net = reload_net()
    outputs = net(Variable(image))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(len(predicted))))

# 测试过程:
# 1. 加载视频并识别双眼
# 2. 调用模型预测是否睁开
def runtest(video_path,filepath):
    # Load the jpg file into a numpy array

    video_capture = cv2.VideoCapture(video_path)
    if video_capture.isOpened():  # 判断是否正常打开
        print("视频读取成功")
    else:
        print("读取视频失败")

    print(video_capture)
    path = osp.join(filepath, 'zy')
    left_eye = osp.join(path, 'left')
    right_eye = osp.join(path, 'right')
    if not osp.exists(left_eye):
        os.makedirs(left_eye)
    if not osp.exists(right_eye):
        os.makedirs(right_eye)

    start = time.time()
    i = 0
    index = 0
    flag = True
    while flag:
        i += 1
        if i > 2:
            break
        flag, image = video_capture.read()
        # image=imresize(image)
        # Find all facial features in all the faces in the image
        # cv2.imshow('a', image)
        face_landmarks_list = face_recognition.face_landmarks(image, model='small')

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
                    left_top = (int(eye_local_1[0] - 0.6 * width), eye_local_1[1] - int(hight * 1.6 / 2))
                    right_bottom = (int(eye_local_2[0] + 0.6 * width), eye_local_1[1] + int(hight * 1.6 / 2))
                    cut_image = None
                    # 裁剪左右眼并调用模型识别是否睁开
                    # cut_image = image[right_bottom[1]:left_top[1], right_bottom[0]:left_top[0]]
                    if facial_feature == 'right_eye':
                        print("处理右眼")
                        try:
                            cut_image = image[right_bottom[1]:left_top[1],right_bottom[0]:left_top[0]]
                            cv2.imshow('a', cut_image)
                            # cv2.imwrite('1.jpg',cut_image)
                            cv2.imwrite(osp.join(right_eye,'{}.jpg'.format(index)), cut_image)
                        except Exception:
                            continue
                    if facial_feature == 'left_eye':
                        print("处理左眼")
                        try:
                            cut_image = image[right_bottom[1]:left_top[1],right_bottom[0]:left_top[0]]
                            cv2.imshow('a', cut_image)
                            # cv2.imwrite('1.jpg',cut_image)
                            cv2.imwrite(osp.join(left_eye,'{}.jpg'.format(index)), cut_image)
                        except Exception:
                            continue
                        cv2.imwrite(osp.join(left_eye,'{}.jpg'.format(index)), cut_image)
                    imgs = Image.fromarray(cut_image.astype('uint8')).convert('RGB')

                    # 将眼镜图片缩放到指定大小并转换成tensor
                    img = transformer(imgs)
                    predict(img)
                    cv2.rectangle(image, left_top, right_bottom, (0, 0, 255), 1)
                    index += 1
                    # for local in local_list:
                    # cv2.circle(image,local,1,(255,255,255),4)

        if i % 20 == 0:
            print("threading {} FPS of the video is {:5.4f}".format(0, i / (time.time() - start)))
        # cv2.imshow('a', image)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "/home/xiaoke/Documents/PPT/zhengyi.avi"
    save_path = "/home/xiaoke/Documents/PPT/faceimage"
    runtest(video_path, save_path)