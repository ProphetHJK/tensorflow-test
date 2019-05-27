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

# 正则表达式
import re

import tensorflow as tf

import cifar100_input

FLAGS = tf.app.flags.FLAGS

# 定义了运行时的控制台参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch的大小""")
tf.app.flags.DEFINE_integer('batch_size_one', 1,
                            """batch大小为1""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """是否使用fp16.""")

# 定义cifar100的导入规格
IMAGE_SIZE = cifar100_input.IMAGE_SIZE
NUM_CLASSES = cifar100_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar100_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar100_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# 定义训练时的参数
MOVING_AVERAGE_DECAY = 0.9999  # 滑动平均衰减.
NUM_EPOCHS_PER_DECAY = 10.0  # 多少轮数后学习率衰减.
LEARNING_RATE_DECAY_FACTOR = 0.96  # 衰减因子 等待注释补充.
INITIAL_LEARNING_RATE = 0.1  # 初始学习率

# 多GPU情况，等待注释补充
TOWER_NAME = 'tower'


def _activation_summary(x):
    """可视化相关，等待注释补充"""
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """在cpu内存中存放变量"""
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """根据参数生成权重，正则化，将权重通过l2正则化加入到loss中"""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def destorted_inputs():
    """导入训练数据"""

    images, labels = cifar100_input.distorted_inputs(batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """导入测试数据"""
    images, labels = cifar100_input.inputs(eval_data=eval_data, batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def single_inputs(eval_data):
    """导入单张图片"""
    images, labels = cifar100_input.inputs(eval_data=eval_data, batch_size=FLAGS.batch_size_one)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inference(images):
    """建立模型

    Args:
        images: inputs()和distorted_input()返回的对象

    """
    with tf.variable_scope('conv1') as scope:
        # 生成64层5*5*3的卷积核
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        # 使用relu激活函数去线性化
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)
    # 第一轮池化
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # 局部响应归一化
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.5,
                      name='norm1')

    with tf.variable_scope('conv2') as scope:
        # 生成64层5*5*3的卷积核
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        # 使用relu激活函数去线性化
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # 局部响应归一化
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # 第二轮池化
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    # 第三层 全连接层
    with tf.variable_scope('local3') as scope:
        # 全部拉直,
        reshape = tf.keras.layers.Flatten()(pool2)
        # 获取tensor的shape长度,此处长度为2304
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # 第四层 全连接层
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # 此处并未进行归一化，因为tf.nn.sparse_softmax_cross_entropy_with_logits函数接受
    # 未经缩放的logits,换句话说该函数能在内部实现缩放操作
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """损失计算，加入L2Loss"""

    # 计算交叉熵
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
    )
    # 平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # 正则化的loss已经加入过了
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """加入图形界面，等待注释"""
    # 滑动平均，decay为0.9
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # 将标量摘要附加到所有个体损失和总损失;
    # 对平均损失做同样的事情。
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """开始训练cifar100模型"""

    # 一轮中的batch数量，也就是一轮训练步数，训练完一个batch为一步
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    # 衰减学习率所需训练步数, 390.625*num_batches_per_epoch
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 根据步数衰减学习率(learning rate),需要修改staitcase为False
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # 生成所有损失的滑动平均值和相关的摘要，此处为什么会需要维护一个
    # 单独的total_loss的影子变量需要注释，可能是为了使最后的曲线较为平滑
    loss_averages_op = _add_loss_summaries(total_loss)

    # 使用梯度下降优化器，需要注释
    # 每次都要先更新影子变量，这里是losses
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)

        # 计算梯度
        grads = opt.compute_gradients(total_loss)
    # 应用梯度，可以直接使用opt.minimize()取代compute_greadients()和apply_gradients()
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 为可训练变量创建直方图
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 梯度直方图
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # 跟踪所有变量的滑动平均值，此时所有变量都不是影子变量，影子变量将会保存起来，在测试数据的时候加载
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        # 应用一次滑动
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op



