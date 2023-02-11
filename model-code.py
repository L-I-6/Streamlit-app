import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


data = np.load('../01.中间数据/data-flower/flo_data.npz')
train_data, train_lab, val_data, val_lab = data['train_data'], data['train_lab'], data['val_data'], data['val_lab']

# 定义训练生成器
train_generator = ImageDataGenerator(
    rotation_range=90,
   zoom_range=0.2,
    rescale=1./255
)
val_generator = ImageDataGenerator(
        rescale=1./255
)
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
def Model():
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
    return model
model = Model()
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
checkpoint= ModelCheckpoint(
    '../../models/model224_2.h5',  # 模型保存的路径  无效路径
    monitor='val_acc',  # 保存模型的条件
    save_best_only=True,  # monitor的判断条件
    save_weight_only=False,  # 是否只保存权重
    verbose=1,  # 训练进度条状态
    mode='min',
    perio=1  # modelcheckpoint之间间隔的epochs数
)

# 模型训练
history = model.fit(
    x=train,
    epochs=epochs,
    callbacks=[checkpoint],
    validation_data=val
)