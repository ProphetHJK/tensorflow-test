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

from datetime import datetime
import time

import tensorflow as tf
import cifar100

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'C:\\cifar100\\checkpoint',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_boolean('load_checkpoint', True,
                            """是否加载检查点继续训练""")


def train():
    """训练cifar100"""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels = cifar100.destorted_inputs()

        # 建立模型，并获取得到的结果logits，用于与labels求交叉熵
        logits = cifar100.inference(images)

        # 计算损失
        loss = cifar100.loss(logits, labels)

        train_op = cifar100.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """打印损失和运行状态"""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')

                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        # MonitoredTrainingSession是一个方便的tensorflow会话初始化/恢复器,
        # 也可用于分布式训练
        with tf.train.MonitoredTrainingSession(
                # 加载保存的训练状态的目录，如为空则设为保存目录
                checkpoint_dir=FLAGS.train_dir,
                # 保存间隔
                save_checkpoint_secs=None,
                save_checkpoint_steps=10000,
                # 可选的SessionRunHook对象列表
                # StopAtStepHook表示停止步数
                # NanTensorHook表示当loss为None时返回异常并停止训练
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=config) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    if not FLAGS.load_checkpoint:
        if tf.gfile.Exists(FLAGS.train_dir):
            # 递归删除文件夹
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
