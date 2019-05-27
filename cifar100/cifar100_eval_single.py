
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from prettytable import PrettyTable  # PrettyTable使用参看http://www.ithao123.cn/content-2560565.html

import cifar100
import numpy as np

FLAGS = tf.app.flags.FLAGS
# 设置存储模型训练结果的路径
tf.app.flags.DEFINE_string('checkpoint_dir', 'C:\\cifar100\\checkpoint',
                           """存储模型训练结果的路径""")
tf.app.flags.DEFINE_string('class_dir', 'C:\\Users\\12567\\tensorflow_datasets\\cifar100\\1.3.1\\label.labels.txt',
                           """存储文件batches.meta.txt的目录""")
tf.app.flags.DEFINE_string('test_file', 'C:\\cifar100\\test.jpg',
                           """测试用的图片""")
IMAGE_SIZE = 24


def evaluate_images(images):  # 执行验证
    logits = cifar100.inference(images)
    load_trained_model(logits=logits)


def load_trained_model(logits):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 从训练模型恢复数据
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        # 下面两行是预测最有可能的分类
        # predict = tf.argmax(logits, 1)
        # output = predict.eval()

        # 从文件以字符串方式获取10个类标签，使用制表格分割
        cifar100_class = np.loadtxt(FLAGS.class_dir, str, delimiter='\t')
        # 预测最大的三个分类
        top_k_pred = tf.nn.top_k(logits, k=3)
        output = sess.run(top_k_pred)
        probability = np.array(output[0]).flatten()  # 取出概率值，将其展成一维数组
        index = np.array(output[1]).flatten()  # 取出索引值，并展开成一维数组，由于只有一张图片实际上只是去掉一层方括号
        # 使用表格的方式显示
        tabel = PrettyTable(["index", "class", "probability"])
        tabel.align["index"] = "l"
        tabel.padding_width = 1
        for i in np.arange(index.size):
            tabel.add_row([index[i], cifar100_class[index[i]], probability[i]])
        print(tabel)


def img_read(filename):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal('File does not exists %s', filename)
    image_data = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(filename),
                                                                   channels=3), dtype=tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    image = tf.image.resize_images(image_data, (height, width), method=ResizeMethod.BILINEAR)
    image = tf.expand_dims(image, -1)
    image = tf.reshape(image, (1, 24, 24, 3))
    return image


def main(argv=None):  # pylint: disable=unused-argument
    filename = FLAGS.test_file
    images = img_read(filename)
    evaluate_images(images)


if __name__ == '__main__':
    tf.app.run()