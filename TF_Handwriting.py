# import packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


# GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# per_process_gpu_memory_fraction
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
config = tf.ConfigProto(gpu_options = gpu_options)
session = tf.Session(config = config)


# define hyper-parameters
# max iterations
max_steps = 5000
# learning rate
learning_rate = 0.001
# dropout percentage
dropout = 0.9
# data directory
data_dir = './MNIST_DATA'
# log directory
log_dir  = './MNIST_LOG'


# download data
mnist = input_data.read_data_sets(data_dir, one_hot = True)

# data processing
# tensorflow setup
sess = tf.InteractiveSession(config = config)
# placeholder setup, x = feature data, y_ = label data
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name = 'y-input')

# record images
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# weight variable initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# bias variable initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# variables summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # calculate mean
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # calculate standard deviation
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        
        # take notes using tf.summary.scalar
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

        # draw a histogram
        tf.summary.histogram('histogram', var)


# Neural Networks setup

# input_tensor = feature data, input_dim = dimention of input 
# output_dim = dimention of output, layer_name = namespace
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
    # namespace setup
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)

        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('linear', preactivate)
        
        activations = act(preactivate, name = 'activation')
        tf.summary.histogram('activations', activations)

    return activations

# hidden layer setup
hidden1 = nn_layer(x, 784, 500, 'layer1')

# dropout layer setup
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# output layer setup
y = nn_layer(dropped, 500, 10, 'layer2', act = tf.identity)


# loss function setup
with tf.name_scope('loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('loss', cross_entropy)


# use AdamOptimizer 
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# calculate accuracy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summary
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# inilize global variables
tf.global_variables_initializer().run()



# feed
def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_:ys, keep_prob: k}

for i in range(max_steps):
    # record summary and accuracy in test set
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict = feed_dict(False))
        test_writer.add_summary(summary, 1)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        # record summary in train set
        summary, _ = sess.run([merged, train_step], feed_dict = feed_dict(True))
        train_writer.add_summary(summary, i)


train_writer.close()
test_writer.close()
