"""本模块用于将一批图片分拣到对应标签文件夹下"""
import tensorflow as tf
import os
import cifar100_eval_single
import cifar100
import numpy as np
import shutil
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'C:\\cifar100\\test_dir',
                           """测试用的图片存放目录""")
IMAGE_SIZE = 27


def evaluate_images(images):  # 执行验证
    logits = cifar100.inference(images)
    return logits


def mkdir_and_copy(pathname, filename):
    """将文件复制到新文件夹种.

    Args:
        pathname: string, 目标文件夹.
        filename: string, 源文件名，在Flags.test_dir下的文件.

    Returns:

    """
    path = os.path.join(FLAGS.test_dir, pathname)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    shutil.copyfile(os.path.join(FLAGS.test_dir, filename), os.path.join(path, filename))


def load_trained_model(logits, image_name_list, batch_size):
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
        cifar100_class = np.loadtxt(FLAGS.class_dir, str, delimiter='\t')
        top_k_pred = tf.nn.top_k(logits, k=1)
        for i in range(batch_size):
            output = sess.run(top_k_pred)
            print(output)
            index = np.array(output[1]).flatten()
            pathname = cifar100_class[index[0]]
            mkdir_and_copy(pathname, image_name_list[i])
            print(cifar100_class[index[0]])


def img_reader(folder):
    """从文件夹读取图片信息.

    Args:
        folder: string, 图片存放目录.

    Returns:
        images_path_list: list, 图片绝对路径列表
        images_name_list: list, 图片名字列表
        size: int, 列表的大小

    """
    file_list = os.listdir(folder)
    images_path_list = []
    images_name_list =[]
    for filename in file_list:
        name = os.path.splitext(filename)[-1]
        if name == '.jpg':
            images_path_list.append(os.path.join(FLAGS.test_dir, filename))
            images_name_list.append(filename)

    size = len(images_name_list)
    return images_path_list, images_name_list, size


def img_processer(image_name_list):
    """将图片名字列表导入Datasets，并处理图片为tensor.

    Args:
        image_name_list: string, 图片名字列表.

    Returns:
        images: Images. 4D tensor of [1, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    """
    images = tf.cast(image_name_list, tf.string)
    # 创建dataset，保存所有图片的路径
    input_queue = tf.data.Dataset.from_tensor_slices(images)
    # 创建迭代器
    iterator = input_queue.make_one_shot_iterator()
    one_image = iterator.get_next()

    image_contents = tf.read_file(one_image)
    images = tf.image.decode_jpeg(image_contents, channels=3)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # image = tf.image.central_crop(image_data, central_fraction=0.7)
    # image = tf.image.resize_image_with_crop_or_pad(image_data, IMAGE_SIZE, IMAGE_SIZE)
    # 标准化图像，便于后面提取特征，(x - mean) / adjusted_stddev等待注释补充，
    #image = tf.image.resize_image_with_pad(image)
    images = tf.image.resize_images(images, (height, width), method=ResizeMethod.BILINEAR)
    # 标准化图像，便于后面提取特征，(x - mean) / adjusted_stddev等待注释补充，
    images = tf.image.per_image_standardization(images)
    # image = tf.expand_dims(image, -1)

    images = tf.reshape(images, (1, 27, 27, 3))

    return images



def main(argv=None):  # pylint: disable=unused-argument
    images_path_list, images_name_list, batch_size = img_reader(FLAGS.test_dir)
    image_list = img_processer(images_path_list)
    logits = evaluate_images(image_list)
    load_trained_model(logits, images_name_list, batch_size)


if __name__ == '__main__':
    tf.app.run()
