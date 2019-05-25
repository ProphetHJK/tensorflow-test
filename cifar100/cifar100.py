from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#正则表达式
import re

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

#定义了运行时的控制台参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch的大小""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """是否使用fp16.""")

# 定义cifar100的导入规格
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL