
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys

tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_sample = mnist.train.num_examples


n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_3 = 512
n_hidden_4 = 10

images = tf.placeholder(tf.float32, shape=[None, 28*28], name='images')
labels = tf.placeholder(tf.int64, shape=[None, 10], name='labels')
ds = tf.contrib.distributions

def weights(shape, Vname):
    initial = tf.truncated_normal(shape, stddev=0.1, name=Vname)
    return tf.Variable(initial)

def bias(shape, Vname):
    bias = tf.constant(0.1, shape=shape, name=Vname)
    return tf.Variable(bias)


def mulitlayer_perceptron (images, weights, bias):

    # First Hidden Layer
    W1 = weights([28* 28, n_hidden_1], 'W1')
    tf.summary.histogram('W1', W1)

    b1 = bias([n_hidden_1], 'b1')
    tf.summary.histogram('b1', b1)

    layer_1 = tf.add(tf.matmul(images, W1), b1)
    tf.summary.histogram('layer1', layer_1)

    layer_1 = tf.nn.relu(layer_1)
    tf.summary.histogram('relu1', layer_1)



    # Second Hidden Layer
    W2 = weights([n_hidden_1, n_hidden_2], 'W2')
    tf.summary.histogram('W2', W2)

    b2 = bias([n_hidden_2], 'b2')
    tf.summary.histogram('b2', b2)

    layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
    tf.summary.histogram('layer2', layer_2)

    layer_2 = tf.nn.relu(layer_2)
    tf.summary.histogram('relu2', layer_2)



    #Third Hidden Layer
    W3 = weights([n_hidden_2, n_hidden_3], 'W3')
    tf.summary.histogram('W3', W3)

    b3 = bias(([n_hidden_3]), 'b3')
    tf.summary.histogram('b3', b3)

    layer_3 = tf.add(tf.matmul(layer_2, W3), b3)
    tf.summary.histogram('Encoder', layer_3)

    # Defining Pz(z)
    mu, rho = layer_3[:, :256], layer_3[:, 256:]
    tf.summary.histogram('mu_z', mu)
    tf.summary.histogram('sigma_z', rho)

    encoding = tf.contrib.distributions.NormalWithSoftplusScale(mu, rho)
    tf.summary.histogram('P(z|x)', encoding.sample())



    # Forth Hidden Layer = Decoder
    W4 = weights([256, n_hidden_4], 'W4')
    tf.summary.histogram('W4', W4)

    b4 = bias([n_hidden_4], 'b4')
    tf.summary.histogram('b4', b4)

    layer_4 = tf.add(tf.matmul(encoding.sample(), W4), b4)
    tf.summary.histogram('Decoder', layer_4)

    return encoding, layer_4


with tf.name_scope('Model'):
    encoding, pred = mulitlayer_perceptron(images, weights, bias)


with tf.name_scope('aproximation_to_prior'):
    prior = tf.contrib.distributions.Normal(0.0, 1.0)
    tf.summary.histogram('r(z)', prior.sample())
    #prior_2 = tf.contrib.distributions.Normal(mu_zy, rho_zy)

with tf.name_scope('Class_Loss'):
    class_loss = tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=labels)

BETA = 0.001

with tf.name_scope('Info_Loss'):
    info_loss = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(encoding, prior), 0)) /math.log(2)

with tf.name_scope('Total_Loss'):
    total_loss = class_loss + BETA * info_loss
    tf.summary.scalar('Total_Loss', total_loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(pred, 1), tf.arg_max(labels, 1)), tf.float32))
tf.summary.scalar('accuracy', accuracy)

IZY_bound = math.log(10, 2) - class_loss
tf.summary.scalar('I(Z;Y)', IZY_bound)

IZX_bound = info_loss
tf.summary.scalar('I(Z;X)', IZX_bound)

batch_size = 100
steps_per_batch = int(mnist.train.num_examples / batch_size)

summary_writer = tf.summary.FileWriter('/home/ali/TensorBoard', graph=tf.get_default_graph())

#Optimization
with tf.name_scope('Optimizer'):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2*steps_per_batch,decay_rate=0.97, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate, 0.5)

    ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
    ma_update = ma.apply(tf.model_variables())

saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt, global_step, update_ops=[ma_update])

tf.global_variables_initializer().run()
merged_summary_op = tf.summary.merge_all()

def evaluate_test():
    IZY, IZX, acc = sess.run([IZY_bound, IZX_bound, accuracy], feed_dict={images: mnist.test.images, labels: mnist.test.labels})
    #print(sig_z)
    return IZY, IZX, acc, 1-acc


import sys

for epoch in range(30):
    for step in range(int(steps_per_batch)):
        im, ls = mnist.train.next_batch(batch_size)
        _, c = sess.run([train_tensor, total_loss], feed_dict={images: im, labels: ls})
        #print(step)
        #summary_writer.add_summary(summary, epoch * steps_per_batch + step)
        #print(mnist.train.num_examples)
        #print(epoch * steps_per_batch + step)
    print("{}: IZY_Test={:.2f}\t IZX_Test={:.2f}\t acc_Test={:.4f}\t err_Test={:.4f}".format(epoch, *evaluate_test()))
    sys.stdout.flush()

savepth = saver.save(sess, '/tmp/mnistvib', global_step)
