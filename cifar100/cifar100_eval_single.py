import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from prettytable import PrettyTable

import cifar100
import numpy as np

FLAGS = tf.app.flags.FLAGS
# 设置存储模型训练结果的路径
tf.app.flags.DEFINE_string('checkpoint_dir', 'C:\\cifar100\\checkpoint',
                           """存储模型训练结果的路径""")
tf.app.flags.DEFINE_string('class_dir', 'C:\\Users\\12567\\tensorflow_datasets\\cifar100\\1.3.1\\coarse_label.labels'
                                        '.txt',
                           """存储文件batches.meta.txt的目录""")
tf.app.flags.DEFINE_string('test_file', 'C:\\cifar100\\test.jpg',
                           """测试用的图片""")
IMAGE_SIZE = 27


def evaluate_images(images):  # 执行验证
    logits = cifar100.inference(images)
    load_trained_model(logits=logits)


def load_trained_model(logits):
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 从训练模型恢复数据
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load success')
        else:
            print('No checkpoint file found')
            return

        # 下面两行是预测最有可能的分类
        # predict = tf.argmax(logits, 1)
        # output = predict.eval()

        # 从文件以字符串方式获取10个类标签，使用制表格分割
        cifar100_class = np.loadtxt(FLAGS.class_dir, str, delimiter='\t')
        # 预测最大的三个分类
        top_k_pred = tf.nn.top_k(logits, k=4)
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
    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(filename),
                                                              channels=3), dtype=tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # image = tf.image.central_crop(image_data, central_fraction=0.7)
    # image = tf.image.resize_image_with_crop_or_pad(image_data, IMAGE_SIZE, IMAGE_SIZE)
    # 标准化图像，便于后面的梯度下降提取特征，(x - mean) / adjusted_stddev等待注释补充，
    #image = tf.image.resize_image_with_pad(image)
    image = tf.image.resize_images(image, (height, width), method=ResizeMethod.BILINEAR)
    image = tf.image.per_image_standardization(image)

    with tf.Session() as sess:
        plt.figure(1)
        plt.imshow(image.eval(session=sess))
        plt.show()
    # image = tf.expand_dims(image, -1)
    image = tf.reshape(image, (1, 27, 27, 3))

    return image


def main(argv=None):  # pylint: disable=unused-argument
    filename = FLAGS.test_file
    images = img_read(filename)
    evaluate_images(images)


if __name__ == '__main__':
    tf.app.run()
