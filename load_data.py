import tensorflow as tf
import numpy as np
import threading
# r = np.arange(0.0,100003.0)
# raw_data = np.dstack((r,r,r,r))[0]
# raw_target = np.array([[1,0,0]] * 100003)
#
# # are used to feed data into our queue
# queue_input_data = tf.placeholder(tf.float32, shape=[20, 4])
# queue_input_target = tf.placeholder(tf.float32, shape=[20, 3])
#
# queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32], shapes=[[4], [3]])
#
# enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
# dequeue_op = queue.dequeue()
#
# # tensorflow recommendation:
# # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
# data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
# # use this to shuffle batches:
# # data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)
#
# def enqueue(sess):
#   """ Iterates over our data puts small junks into our queue."""
#   under = 0
#   max = len(raw_data)
#   while True:
#     print("starting to write into queue")
#     upper = under + 20
#     print("try to enqueue ", under, " to ", upper)
#     if upper <= max:
#       curr_data = raw_data[under:upper]
#       curr_target = raw_target[under:upper]
#       under = upper
#     else:
#       rest = upper - max
#       curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
#       curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
#       under = rest
#
#     sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
#                                     queue_input_target: curr_target})
#     print("added to the queue")
#   print("finished enqueueing")
#
# # start the threads for our FIFOQueue and batch
# sess = tf.Session()
# enqueue_thread = threading.Thread(target=enqueue, args=[sess])
# enqueue_thread.isDaemon()
# enqueue_thread.start()
#
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord, sess=sess)
#
# # Fetch the data from the pipeline and put it where it belongs (into your model)
# for i in range(5):
#   #run_options = tf.RunOptions(timeout_in_ms=4000)
#   curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch])
#   print(curr_data_batch)
#
# # shutdown everything to avoid zombies
# #sess.run(queue.close(cancel_pending_enqueues=True))
# coord.request_stop()
# coord.join(threads)
#sess.close()




# N_SAMPLES = 1000
# NUM_THREADS = 4
# # Generating some simple data
# # create 1000 random samples, each is a 1D array from the normal distribution (10, 1)
# data = 10 * np.random.randn(N_SAMPLES, 4) + 1
# # create 1000 random labels of 0 and 1
# target = np.random.randint(0, 2, size=N_SAMPLES)
#
# queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])
#
# enqueue_op = queue.enqueue_many([data, target])
# data_sample, label_sample = queue.dequeue()
# dequeue_op = queue.dequeue()
# #data_sample, label_sample = tf.train.batch(dequeue_op,capacity=50,batch_size= 5)
# # create ops that do something with data_sample and label_sample
#
# # create NUM_THREADS to do enqueue
# qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
#
# tf.train.add_queue_runner(qr)
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#
#     threads = tf.train.start_queue_runners(coord=coord,sess=sess)
#     for t in threads:
#         print t.name
#     #enqueue_threads = qr.create_threads(sess, coord, start=False)
#     for step in xrange(100):  # do to 100 iterations
#         if coord.should_stop():
#             break
#         data_batch, label_batch = sess.run([data_sample, label_sample])
#     coord.request_stop()
#     coord.join(threads)





















#
# r = np.arange(0.0,1000.0)
# raw_data = np.dstack((r,r,r,r))[0]
# raw_target = np.array([[1,0,0]] * 1000)
#
#
# queue_input_data = tf.placeholder(tf.float32, shape=[20, 4])
# queue_input_target = tf.placeholder(tf.float32, shape=[20, 3])
#
# queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32, tf.float32], shapes=[[4], [3]])
#
# enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
#
#
# dequeue_op = queue.dequeue()
#
# #data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40,min_after_dequeue = 4)
# data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=5, capacity=50,num_threads=3)
#
# # qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)
# # tf.train.add_queue_runner(qr)
#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     under = 0
#     threads = tf.train.start_queue_runners(sess,coord=coord)
#
#
#     for i in range(10):
#
#         upper = under+20
#
#         cur_data = raw_data[under:upper]
#         cur_label = raw_target[under:upper]
#         _,datas,labels = sess.run([enqueue_op,data_batch,target_batch],feed_dict={queue_input_data:cur_data,
#                                                                    queue_input_target:cur_label})
#
#         under = upper
#         print datas
#         print under
#         under += 20
#
#     coord.request_stop()
#     coord.join(threads)


N_SAMPLES = 1000
NUM_THREADS = 7
# Generating some simple data
# create 1000 random samples, each is a 1D array from the normal distribution (10, 1)
data = 10 * np.random.randn(N_SAMPLES, 4) + 1
# create 1000 random labels of 0 and 1
target = np.random.randint(0, 2, size=N_SAMPLES)

with tf.Graph().as_default():
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
    labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

    input_queue = tf.FIFOQueue(capacity=100000,
                                          dtypes=[tf.string, tf.int64],
                                          shapes=[(3,), (3,)],
                                          shared_name=None, name=None)

    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

    num_threads = 4
    data_labels = []
    for _ in xrange(num_threads):
        f,l = input_queue.dequeue()

        data_labels.append([f,l])

    image_batch, labels_batch = tf.train.batch_join(
        data_labels, batch_size=batch_size_placeholder,
        shapes=[(90, 90, 3), ()], enqueue_many=True,
        capacity=4 * 4 * 5,
        allow_smaller_final_batch=True)






