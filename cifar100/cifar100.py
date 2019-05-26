from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#正则表达式
import re

import tensorflow as tf

import cifar100_input

FLAGS = tf.app.flags.FLAGS

#定义了运行时的控制台参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch的大小""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """是否使用fp16.""")

# 定义cifar100的导入规格
IMAGE_SIZE = cifar100_input.IMAGE_SIZE
NUM_CLASSES = cifar100_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar100_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar100_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# 定义训练时的参数
MOVING_AVERAGE_DECAY = 0.9999     # 滑动平均衰减.
NUM_EPOCHS_PER_DECAY = 350.0      # 多少轮数后学习率衰减.
LEARNING_RATE_DECAY_FACTOR = 0.1  # 衰减因子 等待注释补充.
INITIAL_LEARNING_RATE = 0.1       # 初始学习率

# 多GPU情况，等待注释补充
TOWER_NAME = 'tower'

