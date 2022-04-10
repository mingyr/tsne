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
        image = example_parsed['image']
        label = example_parsed['label']

        return image, label
        
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

LOG_DIR = "output-mnist-vector"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

input_ = Input(num_samples, [image_height, image_width, 1], 1)
images, labels = input_("/home/yrming/Data/MNIST/mnist-test.tfr")
labels = tf.squeeze(labels)
    
# 由输入得到模型的输出
def sim(P):
    return tf.map_fn(lambda E: tf.map_fn(lambda F: tf.math.reduce_mean(tf.math.squared_difference(E, F)),
        P, parallel_iterations=16), P, parallel_iterations=16)
        
tsne = TSNE(num_samples)
latents = tsne(images, sim)

# saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())   

def draw_image(sess, v, l):
    @tfmpl.figure_tensor
    def scatter(v, l): 
        '''Draw scatter plots. One for each color.'''
        fig = tfmpl.create_figure(figsize=(8, 6))
        ax = fig.subplots()
        markers = ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd', '.', ',']
        lb_cat = set(l)
        num_cat = len(lb_cat)
        markers = markers[:num_cat]
        for i in range(num_cat):
            lb = lb_cat.pop()
            lb_mask = l == lb
            marker = markers[i]
            data = v[lb_mask]
            x, y = zip(*data)
            ax.plot(x, y, marker, label="{0}".format(lb))

        ax.legend(numpoints=1) 
        fig.tight_layout()

        return fig

    
    image_tensor = scatter(v, l)
    image_summary = tf.summary.image('t-SNE', image_tensor)
    if not sess:
        sess = tf.get_default_session()
    image_str = sess.run(image_summary)

    return image_str


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), 
              tf.local_variables_initializer()])

    v, l = sess.run([latents, labels])
    
    images_str = draw_image(sess, v, l)
    writer.add_summary(images_str, global_step = 0)

    writer.close()
