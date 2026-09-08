#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


import tensorflow.compat.v1 as tf1
# import tensorflow as tf2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.03
session = tf1.Session(config=config)


# In[3]:


train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'


# In[3]:


import tensorflow as tf
import os

# 이미지와 마스크의 경로를 정의합니다.
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'

# 이미지를 로드하고 리사이즈하는 함수입니다.
def load_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # JPEG 형식의 이미지를 디코드합니다.
    image = tf.image.resize(image, [2160, 3840])  # 이미지 크기를 (2160, 3840)으로 조정합니다.
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # PNG 형식의 마스크를 디코드합니다.
    mask = tf.image.resize(mask, [2160, 3840])  # 마스크 크기를 (2160, 3840)으로 조정합니다.
    
    return image, mask

# 데이터셋을 생성하는 함수입니다.
def create_dataset(image_dir, mask_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    mask_paths = [os.path.join(mask_dir, fname.replace('.jpg', '.png')) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)  # 비동기 로딩
    dataset = dataset.batch(8)  # 배치 크기를 8로 설정합니다.
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)  # 데이터셋을 캐시하고 프리페치하여 성능 개선
    return dataset

# 학습 및 검증 데이터셋을 생성합니다.
train_dataset = create_dataset(train_image_dir, train_mask_dir)
val_dataset = create_dataset(val_image_dir, val_mask_dir)


# In[ ]:





# In[7]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

def create_deeplabv3_model(input_shape):
    base_model = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True  # 기본 모델을 훈련 가능하게 설정

    inputs = layers.Input(shape=input_shape)

    # DenseNet121 모델에서 중간 레이어 추가
    x = base_model(inputs)
    # 마지막 레이어 수정: 1채널 출력 (불과 연기 이진 분류)
    x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    # Lambda 레이어를 사용하여 출력 크기 조정
    model_output = layers.Lambda(lambda x: tf.image.resize(x, (2160, 3840)))(x)

    model = keras.Model(inputs=inputs, outputs=model_output)
    return model

# 모델 생성
model = create_deeplabv3_model((2160, 3840, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약 확인
model.summary()


# In[11]:


# 필요한 라이브러리 임포트
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# 데이터셋 생성 (이전 코드 사용)
train_dataset = create_dataset(train_image_dir, train_mask_dir)
val_dataset = create_dataset(val_image_dir, val_mask_dir)

# 모델 생성
model = create_deeplabv3_model((2160, 3840, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 콜백 설정
callbacks = [
    keras.callbacks.ModelCheckpoint("deeplabv3_fire_smoke.h5", save_best_only=True, monitor="val_loss"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
]

# 모델 훈련
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    steps_per_epoch=len(train_dataset),
    validation_steps=len(val_dataset),
    callbacks=callbacks,
    batch_size=2  # 예시로 배치 크기 줄이기
)


# In[ ]:


callbacks = [
    keras.callbacks.ModelCheckpoint("deeplabv3_fire_smoke.h5", save_best_only=True, monitor="val_loss"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
]

model.fit(train_images, train_masks,
          validation_data=(val_images, val_masks),
          epochs=50,
          batch_size=8,  # 배치 크기에 맞게 조정
          callbacks=callbacks)

