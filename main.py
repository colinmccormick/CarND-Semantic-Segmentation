#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """    
    #  Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess,['vgg16'],vgg_path)

    # get the graph and specific tensors from it
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    with tf.variable_scope('FCN8'):
        # do 1x1 convolution after final vgg layer
        FCN08 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # upsample layer (2x) and add skip connection
        FCN09 = tf.layers.conv2d_transpose(FCN08, num_classes, 4, 2, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        VGG04_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        FCN09_skip = tf.add(FCN09, VGG04_1x1)
        # upsample layer (2x) and add skip connection
        FCN10 = tf.layers.conv2d_transpose(FCN09_skip, num_classes, 4, 2, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        VGG03_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        FCN10_skip = tf.add(FCN10, VGG03_1x1)
        # upsample layer (4x)
        FCN11 = tf.layers.conv2d_transpose(FCN10_skip, num_classes, 16, 8, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    return FCN11
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=correct_label))
    # add L2 regularization loss here
    REGULARIZATION_MULTIPLIER = 0.1
    cross_entropy_loss += REGULARIZATION_MULTIPLIER * tf.losses.get_regularization_loss()
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    print("Training...\n")
    for epoch in range(epochs):
        print("Epoch {}...".format(epoch+1))
        total_loss = 0
        for image_batch, label_batch in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                        feed_dict={input_image:image_batch, 
                                    correct_label:label_batch,
                                    keep_prob:0.5,
                                    learning_rate:0.001})
            total_loss += loss
        print("Loss = {:.1f}\n".format(total_loss))
tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        # OPTIONAL: Augment Images for better results
        # set hyper parameters
        epochs = 24
        batch_size = 5
        # build placeholders
        correct_label = tf.placeholder(tf.int32,[None,None,None,num_classes],name='correct_label')
        learning_rate = tf.placeholder(tf.float32,name='learning_rate')
        # define layers, training operation
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess,vgg_path)
        model_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, 
                                                    learning_rate, num_classes)
        # train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
                cross_entropy_loss, input_image,
                correct_label, keep_prob, learning_rate)
        # save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # save model
        #MODEL_PATH = './models/model_01'
        #saver = tf.train.Saver()
        #save_path = saver.save(sess, MODEL_PATH)
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
