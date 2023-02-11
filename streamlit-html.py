import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
# import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

st.set_page_config(
    page_title='图像识别模型-阿强模型',  # 网页标题
    page_icon=':heart:'  # 网页图标
)
st.header('-------------欢迎使用!-阿强模型-------------')
st.subheader('-------------选择好模型在上传照片！-------------')
st.snow()

# 侧边栏
img = plt.imread('./img.jpeg')
st.sidebar.image(img)
st.sidebar.subheader('选择你要使用的模型！')
#模型选择
model_name = st.sidebar.selectbox(label='',options=['花卉识别','天气识别','肺炎检测'])
#重新训练
Retraining = st.sidebar.radio(label='是否重新训练模型(很费时！建议False)',options=['是','否'])
# st.sidebar.write(Retraining)

try:
    img_ = st.file_uploader(label='',)
    st.image(img_)
except:
    st.write('未上传图片')
while (img_!=None ):
    if model_name=='花卉识别':

            if Retraining=='是':
                # ----- 读取数据 -----
                data = np.load('./flo_data.npz')

                train_data, train_lab, val_data, val_lab = data['train_data'], data['train_lab'], data['val_data'], \
                                                           data['val_lab']
                # ----- 定义图像增强生成器 -----
                # 定义训练生成器
                train_generator = ImageDataGenerator(
                    rotation_range=90,
                    zoom_range=0.2,
                    rescale=1. / 255
                )
                val_generator = ImageDataGenerator(
                    rescale=1. / 255
                )
                # 实现图像增强，并批量打包数据（32张图片为一批次）
                train = train_generator.flow(
                    x=train_data,
                    y=train_lab,
                    batch_size=32
                )
                val = val_generator.flow(
                    x=val_data,
                    y=val_lab,
                    batch_size=32
                )
                # ----- 模型搭建 -----

                tf.keras.backend.clear_session()  # 清空模型占用内存
                model = Sequential()
                model.add(Conv2D(64, kernel_size=3, input_shape=(224, 224, 3), activation='relu'))
                model.add(MaxPooling2D())
                model.add(Conv2D(128, kernel_size=3, activation='relu'))
                model.add(MaxPooling2D())
                model.add(Conv2D(128, kernel_size=3, activation='relu'))
                model.add(MaxPooling2D())
                model.add(Conv2D(256, kernel_size=3, activation='relu'))
                model.add(MaxPooling2D())
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(7, activation='softmax'))
                # ----- 模型编译 -----
                lr = 0.0001  # 学习率
                epochs = 10  # 训练次数
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # 自适应矩估计
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # 分类交叉熵损失
                    metrics=['acc']  # 准确率
                )
                # ----- 模型训练 -----
                # 定义检查点
                checkpoint = ModelCheckpoint(
                    'flower_model.h5',  # 模型保存的路径
                    monitor='val_acc',  # 保存模型的条件
                    save_best_only=True,  # monitor的判断条件
                    save_weight_only=False,  # 是否只保存权重
                    verbose=1,  # 训练进度条状态
                    mode='max',
                    perio=1  # modelcheckpoint之间间隔的epochs数
                )

                st.write('训练进度：')
                my_bar = st.progress(0)

                for percent_complete in range(epochs):
                    model.fit(
                        x=train,
                        epochs=1,
                        callbacks=[checkpoint],
                        validation_data=val
                    )
                    my_bar.progress((percent_complete+1)*10)

            else:

                model= tf.keras.models.load_model('flower_model.h5')
            # 预处理
            image=Image.open(img_).convert('RGB')
            image = image.resize((224, 224))
            image = np.array(image)
            image = np.expand_dims(image, axis=0)
            # 模型预测
            pre = model.predict(image).argmax(1)[0]

            dic_flower = {0: 'dandelion',
             1: 'morning_lory',
             2: 'peony',
             3: 'plumeria_ubra',
             4: 'rose',
             5: 'sunflower',
             6: 'tulips'}
            st.write('我认为它是：',dic_flower[pre])
            img_=None
    elif model_name=='天气识别':
        #照片预处理
        if Retraining=='是':
            st.write('还没开发 选False拉重新训练太费时间了！ 默认跳过从新训练')

        from torchvision import transforms,models
        import torch

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        def Transform(image):
            transform = transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image = Image.open(image).convert('RGB')
            image = transform(image)
            image = torch.reshape(image, (1, 3, 244, 244))
            return  image
        #模型加载

        model_weather = models.AlexNet(num_classes=4).cuda()

        model_weather.load_state_dict(torch.load("weather.h5"))
        weather_dic ={0:'cloudy', 1:'rain', 2:'shine', 3:'sunrise'}
        img = Transform(img_)
        st.write('模型认为现在是：', weather_dic[model_weather(img.to(DEVICE)).argmax(1).item()])
        img_ = None
    elif model_name=='肺炎检测':
        #照片预处理
        import torch
        from torchvision import  transforms,models
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        if Retraining=='是':
            st.write('别闹现成模型不香？重新训练也是一样的还费时间 我给你跳过')
        def Tranform(image):
            transform = transforms.Compose([
                transforms.RandomResizedCrop(300, scale=(0.8, 1.1)),
                transforms.CenterCrop(244),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image = Image.open(image).convert('RGB')
            image = transform(image)
            image = torch.reshape(image, (1, 3, 244, 244))
            return image
        # 模型加载
        model_pneumonia = models.AlexNet(num_classes=2).cuda()
        model_pneumonia.load_state_dict(torch.load("pneumonia.h5"))
        Result = {0: '没有', 1: '有'}
        img = Tranform(img_)
        st.write('模型认为你{}肺炎'.format( Result[model_pneumonia(img.to(DEVICE)).argmax(1).item()]))
        img_ = None