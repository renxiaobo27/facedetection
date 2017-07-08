import tensorflow as tf
import numpy as np

#multiple thread read data
N_SAMPLES = 1000
NUM_THREADS = 7
# Generating some simple data
# create 1000 random samples, each is a 1D array from the normal distribution (10, 1)
data = 10 * np.random.randn(N_SAMPLES, 4) + 1
# create 1000 random labels of 0 and 1
target = np.random.randint(0, 2, size=N_SAMPLES)

queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

enqueue_op = queue.enqueue_many([data, target])
#data_sample, label_sample = queue.dequeue()
dequeue_op = queue.dequeue()
data_sample, label_sample = tf.train.batch(dequeue_op,capacity=50,batch_size= 5)
# create ops that do something with data_sample and label_sample

# create NUM_THREADS to do enqueue
qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)

tf.train.add_queue_runner(qr)
with tf.Session() as sess:
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    for t in threads:
        print t.name
    #enqueue_threads = qr.create_threads(sess, coord, start=False)
    for step in xrange(100):  # do to 100 iterations
        if coord.should_stop():
            break
        data_batch, label_batch = sess.run([data_sample, label_sample])
    coord.request_stop()
    coord.join(threads)