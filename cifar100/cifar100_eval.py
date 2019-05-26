from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar100

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'C:\\Users\\Jinkai\\Source\\python\\cifar10\\cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'C:\\Users\\Jinkai\\Source\\python\\cifar10\\cifar',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def eval_once(saver, summary_writer, top_k_op, summary_op):
    """运行一次验证测试"""
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 加载检查点数据
            saver.restore(sess, ckpt.model_checkpoint_path)

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            # 加载未成功
            print('No checkpoint file found')
            return


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # 导入cifar100中的测试数据
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar100.inputs(eval_data=eval_data)

    # 通过卷积神经网络得到结果
    logits = cifar100.inference(images)

    # 每个logits元素前k个最大值是否包含labels的正确结果
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # 恢复学习完成后的滑动平均变量
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar100.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    # 将保存的影子变量值直接赋予当前变量，等待注释
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()