import tensorflow as tf
import numpy as np
from math import cos, sin, pi
import argparse
from tensorflow.python.framework import ops
import lme_custom_ops
import pyconrad as pyc
pyc.setup_pyconrad()
import os
import time
from PIL import Image

class CheckerDrawer:

    # create a numpy array of zeros and ones
    def draw(self, resolution, tile_size):
        width = int(resolution / (tile_size * 2))
        return np.vstack(width * (width * [0, 1], width * [1, 0]))

# ===================== Volume Parameters =====================
class Volume_Params:

    def trajectory(self, number_of_projections, circle):
        rays = np.zeros([number_of_projections, 2])
        angular_increment = circle / number_of_projections
        for i in np.arange(0, number_of_projections):
            rays[i] = [cos(i * angular_increment), sin(i * angular_increment)]
        return rays

    def __init__(self):
        # Volume Parameter:
        self.volume_dim = 256
        self.volume_shape = [self.volume_dim, self.volume_dim]
        self.volume_spacing = 0.5

        # Detector Parameter
        self.detector_width = 512
        self.detector_spacing = 0.5

        # Trajectory Parameter
        self.number_of_projections = 512
        self.circle = 2 * pi

        # Tensor Proto Stuff
        self.volume_origin = tf.contrib.util.make_tensor_proto([-((self.volume_dim - 1) / 2 * self.detector_spacing),
                                                           -((self.volume_dim - 1) / 2 * self.detector_spacing)],
                                                          tf.float32)

        self.volume_spacing = tf.contrib.util.make_tensor_proto([self.volume_spacing,
                                                            self.volume_spacing],
                                                           tf.float32)

        self.sinogram_shape = [self.number_of_projections, self.detector_width]

        self.detector_origin = tf.contrib.util.make_tensor_proto([-((self.detector_width - 1) / 2 * self.detector_spacing)],
                                                            tf.float32)
        self.detector_spacing = tf.contrib.util.make_tensor_proto([self.detector_spacing], tf.float32)

        self.ray_vectors = tf.contrib.util.make_tensor_proto(self.trajectory(self.number_of_projections, self.circle), tf.float32)

@ops.RegisterGradient( "ParallelBackprojection2D" )
def _backproject_grad( op, grad ):
    volume_params = Volume_Params()
    proj = lme_custom_ops.parallel_projection2d(
            volume                      = grad,
            volume_shape                = volume_params.volume_shape,
            projection_shape            = volume_params.sinogram_shape,
            volume_origin               = volume_params.volume_origin,
            detector_origin             = volume_params.detector_origin,
            volume_spacing              = volume_params.volume_spacing,
            detector_spacing            = volume_params.detector_spacing,
            ray_vectors                 = volume_params.ray_vectors,
        )
    return [ proj ]

def generateSinogram(phantom, sino_sess):
    # Create VolumeParameters
    volume_params = Volume_Params()
    # TF Phantom Var
    phantom_tf = tf.placeholder(tf.float32, shape=volume_params.volume_shape, name="input_phantom")

    # TF Layer Object
    forwardprojection_layer = lme_custom_ops.parallel_projection2d(phantom_tf,
                                                                     volume_params.volume_shape,
                                                                     volume_params.sinogram_shape,
                                                                     volume_params.volume_origin,
                                                                     volume_params.detector_origin,
                                                                     volume_params.volume_spacing,
                                                                     volume_params.detector_spacing,
                                                                     volume_params.ray_vectors)

    # ===================== TF Session =====================

    # TF STUFF
    init_op = tf.global_variables_initializer()
    sino_sess.run(init_op)

    # just do forward projection
    sinogram = sino_sess.run(forwardprojection_layer, feed_dict={phantom_tf: phantom})

    return sinogram


def main():

    # Create VolumeParameters
    volume_params = Volume_Params()

    #data2 = np.load("data.npy")
    raw_data = np.load("volumes.npy")
    data = np.zeros((raw_data.shape[2], raw_data.shape[0], raw_data.shape[1]))
    for i in range(raw_data.shape[2]):
        data[i,:,:] = raw_data[:,:,i]
    print("data2", data.shape)
    train_phantoms = data[0:50]
    test_phantoms = data[51:100]
    np.save("./reconst/ground_truth.npy", test_phantoms[0])
    # append some noise to train data
    #train_phantoms = np.append(train_phantoms, np.random.uniform(size=(10,volume_params.volume_dim,volume_params.volume_dim)), axis=0)
    #train_phantoms = np.random.uniform(size=(10,volume_params.volume_dim,volume_params.volume_dim))

    # generate sinograms of training data
    train_sinograms = np.zeros((train_phantoms.shape[0],)+tuple(volume_params.sinogram_shape))
    test_sinograms = np.zeros((test_phantoms.shape[0],)+tuple(volume_params.sinogram_shape))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sino_sess:
        for i in range(train_phantoms.shape[0]):
            train_sinograms[i] = generateSinogram(train_phantoms[i], sino_sess)
        for i in range(test_phantoms.shape[0]):
            test_sinograms[i] = generateSinogram(test_phantoms[i], sino_sess)

    # checkerboard phantoms
    checker_drawer = CheckerDrawer()
    check = checker_drawer.draw(256, 1)

    # Get Phantom
    conrad_phantom_class = pyc.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
    phantom = conrad_phantom_class.SheppLogan(volume_params.volume_dim, False).as_numpy()

    # random phantom
    rand_phant = np.random.uniform(size=(volume_params.volume_dim, volume_params.volume_dim))
    rand_phant2 = np.random.normal(size=(volume_params.volume_dim, volume_params.volume_dim))


    #_______________ Build Network _______________
    # TF Reco Var
    sinogram_tf = tf.placeholder(tf.float32, shape=volume_params.sinogram_shape, name="input_sinogram")

    #fft
    fft_layer = tf.cast(tf.spectral.fft(tf.cast(sinogram_tf,dtype=tf.complex64)),tf.complex64)
    print("fft done")

    #tensorflow multiplication layer
    frequencies = np.fft.fftfreq(n=volume_params.detector_width,d=1)
    fourier_filter = np.abs(frequencies)

    #filter_weights = tf.Variable(tf.convert_to_tensor(fourier_filter,dtype=tf.float32))
    filter_weights = tf.Variable(tf.ones((volume_params.detector_width),dtype=tf.float32))
    
    filter_layer = tf.multiply(fft_layer, tf.cast(filter_weights,dtype=tf.complex64))
    print("filter done")

    #ifft
    ifft_layer = tf.cast(tf.spectral.ifft(tf.cast(filter_layer,dtype=tf.complex64)),dtype=tf.float32)
    print("ifft done")


    # reconstruct phantom again
    backprojection_layer = lme_custom_ops.parallel_backprojection2d( sinogram=ifft_layer,
                                                    sinogram_shape=volume_params.sinogram_shape,
                                                    volume_shape=volume_params.volume_shape,
                                                    volume_origin=volume_params.volume_origin,
                                                    detector_origin=volume_params.detector_origin,
                                                    volume_spacing=volume_params.volume_spacing,
                                                    detector_spacing=volume_params.detector_spacing,
                                                    ray_vectors=volume_params.ray_vectors )
    print("backprojection_layer done")

    ground_truth_tf = tf.placeholder(tf.float32, shape=volume_params.volume_shape, name="ground_truth")
    loss_fkt = tf.losses.mean_squared_error(ground_truth_tf, backprojection_layer)

    learning_rate = 1e-5
    epochs = 10000
    g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = g_opt.minimize(loss_fkt)

    # ===================== TF Session =====================
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, "./model.ckpt")
        reco = sess.run(backprojection_layer, feed_dict={sinogram_tf: test_sinograms[0]})
        np.save("./reconst/imgfinal.npy", reco)
        assert False, 'blablabla'
        reco = None
        train_losses = []
        test_losses = []
        print("start training loop")
        for epoch in range(epochs):
            phantom = None
            sinogram = None
            #idx = np.random.randint(0, train_phantoms.shape[0])
            for it in range(train_sinograms.shape[0]):
                sinogram = train_sinograms[it]
                phantom = train_phantoms[it]
                training = sess.run(train_op, feed_dict={sinogram_tf: sinogram, ground_truth_tf: phantom})
            # run tf session
            if epoch%50 == 0 or epoch == 0:
                filter_values = sess.run(filter_weights, feed_dict={sinogram_tf: sinogram, ground_truth_tf: phantom})
                np.save("./filters/f"+str(epoch+1),filter_values)
                train_loss_value = sess.run(loss_fkt, feed_dict={sinogram_tf: sinogram, ground_truth_tf: phantom})
                print("epoch: ", epoch,"train loss value: ", train_loss_value)
                idx = np.random.randint(0, test_phantoms.shape[0])
                test_loss_value = sess.run(loss_fkt, feed_dict={sinogram_tf: test_sinograms[idx], ground_truth_tf: test_phantoms[idx]})
                print("epoch: ", epoch,"test loss value: ", test_loss_value)
                reco = sess.run(backprojection_layer, feed_dict={sinogram_tf: test_sinograms[0]})
                np.save("./reconst/img"+str(epoch+1)+".npy", reco)
                train_losses.append(train_loss_value)
                test_losses.append(test_loss_value)
        np.savetxt("train_losses.csv", train_losses, delimiter=",")
        np.savetxt("test_losses.csv", test_losses, delimiter=",")

        # save model
        save_path = saver.save(sess, "./model.ckpt")

        # run tf session, to get reco
        pyc.imshow(reco, 'label reco')



if __name__ == '__main__':
    main()