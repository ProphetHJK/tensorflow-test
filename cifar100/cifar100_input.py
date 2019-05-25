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

class DataPreprocessor(object):


def _get_images_labels(batch_size, split, distords=False)
    """获取标签信息，返回dataset"""
    # 导入tfds支持的数据集,split为拆分方式
    dataset = tf.load(name='cifar100', split=split)
    scope = 'data_augmentation' if distords else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
    dataset = dataset.prefech(-1)
    dataset = dataset.repeat().batch(batch_size)
    # 创建一个迭代器
    iterator = dataset.make_one_shot_iterator()
    images_labels = iterator.get_next()
    images, labels = images_labels['']