import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# === prepare tensorflow network === #

#config setting
imageDim = 784
outputDim = 10

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# None means that a dimension can be any of any length
x = tf.placeholder(tf.float32, [None, imageDim], name="input")
# 784-dimensional image vectors by it to produce 10-dimensional vectors
W = tf.Variable(tf.zeros([imageDim, outputDim]), dtype=tf.float32, name="Weight")
# a shape of [10]
b = tf.Variable(tf.zeros([outputDim]), dtype=tf.float32, name="bias")
# softmax
y = tf.nn.softmax(tf.matmul(x, W)+b, name="softmax")
print(x, W, b, y)

# input correct answers
y_ = tf.placeholder(tf.float32, [None, outputDim])

# cross-entropy
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

print("tensorflow network already prepare done...")
print("session start run...")
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
#Training
    for i in range(1000):
        if i % 100 == 0:
            print("iteration num :", i)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})

    # Save checkpoint, graph.pb and tensorboard
    saver.save(sess, "models/model.ckpt")
    tf.train.write_graph(sess.graph.as_graph_def(), "models/", "graph.pb")
    tf.summary.FileWriter("board", sess.graph)
    print("ckpt model save in the file : models/model.ckpt")
    print("graph model save in the file : models/graph.pb")
    print("board model save in the directory : board/")

#Testing
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(
        "accuracy : ",    
        sess.run(
            accuracy,
            feed_dict={x: mnist.test.images, y_:mnist.test.labels}
        )
     )

