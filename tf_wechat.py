import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

from itchat_util import *

# 初始化参数
param = (0.001, 20000, 128, 10)

wechat = WeChatSendMsg()


def nn_train():
    # mnist data reading
    mnist = input_data.read_data_sets("data/", one_hot=True)

    # Parameters
    # learning_rate = 0.001
    # training_iters = 200000
    # batch_size = 128
    # display_step = 10
    # learning_rate, steps_per_epoch, batch_size, display_step = param
    learning_rate = wechat.get_param()['learning_rate']
    epochs = wechat.get_param()['epochs']
    steps_per_epoch = wechat.get_param()['steps_per_epoch']
    batch_size = wechat.get_param()['batch_size']
    display_step = 10

    args = 'learning_rate: ' + str(learning_rate) + \
           '\nepochs: ' + str(epochs) + \
           '\nsteps_per_epoch: ' + str(steps_per_epoch) + \
           '\nbatch_size: ' + str(batch_size)
    wechat.send(args)

    # Network Parameters
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)
    dropout = 0.75  # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        # print('Wait for lock')
        # with lock:
        #     run_state = running
        # print('Start')
        run_state = wechat.get_run_state()
        while step * batch_size < steps_per_epoch and run_state:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                msg = ("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +
                       ", Training Accuracy= " + "{:.5f}".format(acc))
                print(msg)
                # to_user_names = ['李西康', '李嵚锋']
                wechat.send(msg)
            step += 1
            run_state = wechat.get_run_state()
        end_msg = "Optimization Finished!"
        print(end_msg)
        wechat.send(end_msg)

        # Calculate accuracy for 256 mnist test images
        acc_msg = "Testing Accuracy:" + \
                  str(sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                                    y: mnist.test.labels[:256],
                                                    keep_prob: 1.}))
        print(acc_msg)
        wechat.send(acc_msg)

    wechat.close()


if __name__ == '__main__':
    wechat.run(nn_train)
