import os
import numpy as np
import tensorflow as tf
import sonnet as snt
import tfmpl

from tsne import TSNE

class Input(snt.AbstractModule):
    def __init__(self, batch_size, image_dims, num_epochs = -1, name = 'input'):
        super(Input, self).__init__(name = name)
        self._batch_size = batch_size
        self._image_dims = image_dims
        self._num_epochs = num_epochs

    def _parse_function(self, example):
        dims = np.prod(self._image_dims)

        features = {
            "image": tf.FixedLenFeature([dims], dtype = tf.float32),
            "label": tf.FixedLenFeature([], dtype = tf.int64)
        }

        example_parsed = tf.parse_single_example(serialized = example, features = features)
        value = tf.reshape(example_parsed['image'], self._image_dims)

        label = example_parsed['label']

        return value, label
        
    def _build(self, filename):
        assert os.path.isfile(filename), "invalid file name: {}".format(filename)

        dataset = tf.data.TFRecordDataset([filename])
        dataset = dataset.map(self._parse_function)

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(self._num_epochs)

        it = dataset.make_one_shot_iterator()
        images, labels = it.get_next()

        return images, labels


image_width = 28
image_height = 28
num_samples = 100

LOG_DIR = "output-mnist-image"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

input_ = Input(num_samples, [image_height, image_width, 1], 1)
images, _ = input_("D:\\Data\\MNIST\\mnist-test.tfr")
    
# 由输入得到模型的输出

def ssim(P):
    return tf.map_fn(lambda E: tf.map_fn(lambda F: 1 - tf.image.ssim(E, F, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03),
        P, parallel_iterations=16), P, parallel_iterations=16)
        
tsne = TSNE(num_samples)
latents = tsne(images, ssim)

# saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())   

def draw_image(sess, v, color):
    @tfmpl.figure_tensor
    def scatter(v, color): 
        '''Draw scatter plots. One for each color.'''  
        fig = tfmpl.create_figures(1, figsize=(8,8))[0]
        ax = fig.add_subplot(111)
        ax.scatter(v[:, 0], v[:, 1], c=color)
        fig.tight_layout()

        return fig

    image_tensor = scatter(v, color)
    image_summary = tf.summary.image('t-SNE', image_tensor)
    if not sess:
        sess = tf.get_default_session()
    image_str = sess.run(image_summary)

    return image_str


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), 
              tf.local_variables_initializer()])

    v = sess.run(latents)
    
    images_str = draw_image(sess, v, 'r')
    writer.add_summary(images_str, global_step = 0)

    writer.close()