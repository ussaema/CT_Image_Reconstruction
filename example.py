import tensorflow as tf
import numpy as np
from math import cos, sin, pi
import argparse
from tensorflow.python.framework import ops
import lme_custom_ops
import pyconrad as pyc
pyc.setup_pyconrad()

class reco_helper:
    @staticmethod
    def trajectory(number_of_projections, circle):
        rays = np.zeros([number_of_projections, 2])
        angular_increment = circle / number_of_projections
        for i in np.arange(0, number_of_projections):
            rays[i] = [cos(i * angular_increment), sin(i * angular_increment)]
        return rays
    @staticmethod
    def generate_sinograms(phantom, args, sess):
        phantom_tf = tf.placeholder(tf.float32, shape=args.volume_shape, name="input_phantom")
        rays = tf.contrib.util.make_tensor_proto(reco_helper.trajectory(args._number_of_projections, args._circle), tf.float32)

        result = lme_custom_ops.parallel_projection2d(phantom_tf, args.volume_shape, args.sinogram_shape,
                                                      args.volume_origin, args.detector_origin,
                                                      args.volume_spacing, args.detector_spacing, rays)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        sinogram = sess.run(result, feed_dict={phantom_tf: phantom})

        return sinogram


class geometry:
    # Volume Parameter
    _volume_xy = 256
    _volume_spacing = 0.5
    # Detector Parameter
    _detector_width = 615  # 725#365
    _detector_spacing = 0.5
    # Trajectory Parameter
    _number_of_projections = 360
    _circle = 2 * pi
    # geometry paramter
    source_detector_distance = 1200
    source_isocenter_distance = 900

    volume_shape = [_volume_xy, _volume_xy]
    volume_origin = tf.contrib.util.make_tensor_proto(
        [-((_volume_xy - 1) / 2 * _volume_spacing), -((_volume_xy - 1) / 2 * _volume_spacing)], tf.float32)
    volume_spacing = tf.contrib.util.make_tensor_proto([_volume_spacing, _volume_spacing], tf.float32)

    sinogram_shape = [_number_of_projections, _detector_width]
    detector_origin = tf.contrib.util.make_tensor_proto([-((_detector_width - 1) / 2 * _detector_spacing)], tf.float32)
    detector_spacing = tf.contrib.util.make_tensor_proto([_detector_spacing], tf.float32)
    rays_tensor = tf.contrib.util.make_tensor_proto(reco_helper.trajectory(_number_of_projections, _circle), tf.float32)


@ops.RegisterGradient( "ParallelProjection2D" )
def _project_grad( op, grad ):
    reco = lme_custom_ops.parallel_backprojection2d(
            sinogram                      = grad,
            sinogram_shape              = op.get_attr("projection_shape"),
            volume_shape                = op.get_attr( "volume_shape" ),
            volume_origin               = op.get_attr( "volume_origin" ),
            detector_origin             = op.get_attr( "detector_origin" ),
            volume_spacing              = op.get_attr( "volume_spacing" ),
            detector_spacing            = op.get_attr( "detector_spacing" ),
            ray_vectors                 = op.get_attr( "ray_vectors" ),
        )
    return [ reco ]

class iterative_reco_model:

    def __init__(self, reco_setup, reco_initialization):
        self.sinogram_shape = reco_setup.sinogram_shape
        self.volume_shape = reco_setup.volume_shape
        self.volume_origin = reco_setup.volume_origin
        self.volume_spacing = reco_setup.volume_spacing
        self.detector_origin = reco_setup.detector_origin
        self.detector_spacing = reco_setup.detector_spacing
        self.rays_tensor = reco_setup.rays_tensor
        self.reco = tf.get_variable(name='reco', dtype=tf.float32, initializer=tf.expand_dims(reco_initialization,axis=0),
                                    trainable=True,constraint=lambda x: tf.clip_by_value(x, 0, np.infty))


    def model(self, input_volume):

        self.updated_reco = tf.add(input_volume,self.reco)

        self.current_sino = lme_custom_ops.parallel_projection2d(self.updated_reco, self.volume_shape, self.sinogram_shape, self.volume_origin,
                                                 self.detector_origin, self.volume_spacing, self.detector_spacing,
                                                 self.rays_tensor)
        return self.current_sino, self.reco

    def abs_criterion(self, in_, target):
        return tf.reduce_mean(tf.abs(in_ - target))

    def l2_norm_mean(selfself, in_, target):
        return tf.reduce_mean(tf.squared_difference(in_,target))

    def l2_norm_sum(selfself, in_, target):
        return tf.reduce_sum(tf.squared_difference(in_,target))


class pipeline(object):

    def bronze_queue(self, inputs, labels, args, shuffle=False):
        """Load the image data in tf.Dataset format and create data queue.
        Feeds (data) faster than your average bronze teammates.
            Args:
              inputs & labels: List of input and label data. Input[i] is expected to correspond to label[i].
            Returns:
               Returns a nested structure of `tf.Tensor`s containing the next element.
            """
        # Make pairs of elements. (X, Y) => ((x0, y0), (x1)(y1)),....
        image_set = tf.data.Dataset.from_tensor_slices((inputs, labels))
        # Identity mapping operation is needed to include multi-tthreaded queue buffering.
        image_set = image_set.map(lambda x, y: (x, y), num_parallel_calls=4).prefetch(buffer_size=200)
        # Shuffle dataset (draw uniform random sample from buffer_size). Position of shuffle relative to batch is important.
        # Before: shuffle all elements, after: shuffle only batches, keep elements in batch fixed.
        if shuffle: image_set = image_set.shuffle(buffer_size=args.train_size)
        # Batch dataset. Also do this if batchsize==1 to add the mandatory first axis for the batch_size
        image_set = image_set.batch(1)
        # Repeat dataset for number of epochs
        image_set = image_set.repeat(args.num_epochs + 1)
        # Prefetch data to gpu. Now feeding faster than your bronze teammates. Must be last transformation in Dataset pipeline.
        # Select iterator
        iterator = image_set.make_initializable_iterator()
        return iterator

    def __init__(self, session, args, volume_shape):
        self.args = args
        self.sess = session
        self.model = iterative_reco_model(geometry, np.zeros([volume_shape, volume_shape], dtype=np.float32))


    def init_placeholder_graph(self):
        self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, initializer=tf.constant(0.0001), trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate_placeholder')
        self.set_learning_rate = self.learning_rate.assign(self.learning_rate_placeholder)

        self.is_training = tf.get_variable(name="is_training", dtype=tf.bool, trainable=False, initializer=True)
        self.set_training = self.is_training.assign(True)
        self.set_validation = self.is_training.assign(False)

    def build_graph(self, input_type, input_shape, label_shape):

        self.init_placeholder_graph()
        g_opt = tf.train.AdamOptimizer(self.learning_rate)

        # Tensor placeholders that are initialized later. Input and label shape are assumed to be equal
        self.inputs_train = tf.placeholder(input_type, (None,input_shape[0], input_shape[1]))
        self.labels_train = tf.placeholder(input_type, (None,label_shape[0], label_shape[1]))
        self.inputs_test = tf.placeholder(input_type, (None,input_shape[0], input_shape[1]))
        self.labels_test = tf.placeholder(input_type, (None,label_shape[0], label_shape[1]))

        # Get next_element-"operator" and iterator that is initialized later
        self.iterator_train = self.bronze_queue(self.inputs_train, self.labels_train, self.args, shuffle=False)
        self.iterator_test = self.bronze_queue(self.inputs_test, self.labels_test, self.args, shuffle=False)

        # Get next (batch of) element pair(s)
        self.input_element, self.label_element = tf.cond(self.is_training,
                                                         lambda: self.iterator_train.get_next(),
                                                         lambda: self.iterator_test.get_next())

        self.current_sino, self.reco_diff = self.model.model(self.input_element)
        self.loss = self.model.l2_norm_sum(self.label_element, self.current_sino)
        self.train_op = g_opt.minimize(self.loss)

        # Summaries and stuff for Tensorboard
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, '')
        summaries.append(tf.summary.scalar('avg_g_loss', tf.reduce_mean(self.loss)))
        self.summary_op = tf.summary.merge(summaries)

    def train(self, input_train, labels_train, inputs_test, labels_test):
        print("len(inputs_train)%f" % len(input_train))
        self.build_graph(input_train.dtype, input_train.shape, labels_train.shape)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=100)

        learning_rate = self.args.learning_rate

        inputs_train = np.expand_dims(input_train, axis=0)
        labels_train = np.expand_dims(labels_train, axis=0)
        inputs_test = np.expand_dims(inputs_test, axis=0)
        labels_test = np.expand_dims(labels_test, axis=0)

        self.sess.run(self.iterator_train.initializer, feed_dict={self.inputs_train: inputs_train, self.labels_train: labels_train})
        self.sess.run(self.iterator_test.initializer,  feed_dict={self.inputs_test: inputs_test, self.labels_test: labels_test})
        min_loss = 1000000000000
        min_loss_reco = None
        for epoch in range(1, self.args.num_epochs + 1):
            _ = self.sess.run([self.set_training, self.set_learning_rate], feed_dict={self.learning_rate_placeholder: learning_rate})

            for step in range(0, len(inputs_train)):
                    _, loss, current_sino, reco_diff = self.sess.run([self.train_op, self.loss, self.current_sino, self.reco_diff])

            if loss > min_loss * 1.005:
                print('training finished')
                break
            if epoch % 50 is 0 :
                print('Epoch: %d'%epoch)
                print('Loss %f'%loss)
                #pyc.imshow(reco_diff,'Phantom: Current Reco')
            if min_loss > loss:
                min_loss = loss
                min_loss_reco = reco_diff
        return min_loss_reco

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-2, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='num_epochs', type=int, default=50000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=4, help='# images used to train')
args = parser.parse_args()

size = 256
_ = pyc.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
phantom = _.SheppLogan(size, False).as_numpy()
#pyc.imshow(phantom,'Phantom: GT')
reco = np.zeros(np.shape(phantom), dtype=np.float32)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sinogram = reco_helper.generate_sinograms(phantom, geometry, sess)
    #pyc.imshow(sinogram, 'label sino')
    zero_vector = np.zeros(np.shape(phantom), dtype=np.float32)

    iter_pipeline = pipeline(sess, args, geometry._volume_xy)
    result = iter_pipeline.train(zero_vector, np.asarray(sinogram), zero_vector, np.asarray(sinogram))
    pyc.to_conrad_grid(result).save_tiff("iter_reco_result.tiff")
