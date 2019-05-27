# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds  # tensorflow官方数据集模块

# 将导入图像处理成的大小
IMAGE_SIZE = 24

NUM_CLASSES = 100  # CIFAR100拥有100个类别
# 训练集50000张，验证集10000张
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _get_images_labels(batch_size, split, distords=False):
    """获取标签信息，返回dataset"""
    # 导入tfds支持的数据集,split为拆分方式
    dataset = tfds.load(name='cifar100', split=split)
    print(dataset)
    scope = 'data_augmentation' if distords else 'input'
    # 等待注释补充
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size)
    # 创建一个迭代器
    iterator = dataset.make_one_shot_iterator()
    images_labels = iterator.get_next()
    images, labels = images_labels['input'], images_labels['target']
    # 可视化张量图像
    tf.summary.image('image', images)
    return images, labels


# 模型预处理，剪裁失真等
class DataPreprocessor(object):
    """模型预处理，剪裁失真等"""

    # distords参数用于表示训练(True)或验证(False)
    def __init__(self, distords):
        self._distords = distords

    # 使类实例变成可调用对象
    def __call__(self, record):
        """处理图片"""
        # 关于record等待注释补充
        img = record['image']
        img = tf.cast(img, tf.float32)
        if self._distords:  # 训练部分
            # 随机剪裁
            img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
            # 随机左右翻转
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        else:  # 为验证集做处理
            # 中心剪裁,不够黑色填充
            img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
        # 标准化图像，便于后面的梯度下降提取特征，(x - mean) / adjusted_stddev等待注释补充，
        img = tf.image.per_image_standardization(img)
        return dict(input=img, target=record['label'])


def inputs(eval_data, batch_size):
    """导入数据集，可选择导入训练部分或验证部分

    Args:
      eval_data: bool, 训练或验证
      batch_size: batch的大小

    Returns:
      images: Images. 4D tensor，size为 [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
      labels: Labels. 1D tensor，size为[batch_size] .
    """
    split = tfds.Split.TEST if eval_data == 'test' else tfds.Split.TRAIN
    return _get_images_labels(batch_size, split)


def distorted_inputs(batch_size):
    """直接导入训练数据"""
    return _get_images_labels(batch_size, tfds.Split.TRAIN, distords=True)
