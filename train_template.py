import argparse
import numpy as np
import time
import sys
import os.path
import tensorflow as tf
from datetime import datetime
import models
import threading




def main(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # store some git version in a text file in log directory

    #Load Training Data

    r = np.arange(0.0, 100003.0)
    raw_data = np.dstack((r, r, r, r))[0]
    raw_target = np.array([[1, 0, 0]] * 100003)

    # are used to feed data into our queue
    queue_input_data = tf.placeholder(tf.float32, shape=[20, 4])
    queue_input_target = tf.placeholder(tf.float32, shape=[20, 3])

    queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32], shapes=[[4], [3]])

    enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
    dequeue_op = queue.dequeue()

    def enqueue(sess):
        """ Iterates over our data puts small junks into our queue."""
        under = 0
        max = len(raw_data)
        while True:
            print("starting to write into queue")
            upper = under + 20
            print("try to enqueue ", under, " to ", upper)
            if upper <= max:
                curr_data = raw_data[under:upper]
                curr_target = raw_target[under:upper]
                under = upper
            else:
                rest = upper - max
                curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
                curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
                under = rest

            sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
                                            queue_input_target: curr_target})
            print("added to the queue")
        print("finished enqueueing")

    # start the threads for our FIFOQueue and batch
    sess = tf.Session()
    enqueue_thread = threading.Thread(target=enqueue, args=[sess])
    enqueue_thread.isDaemon()
    enqueue_thread.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Fetch the data from the pipeline and put it where it belongs (into your model)
    for i in range(5):
        # run_options = tf.RunOptions(timeout_in_ms=4000)
        curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch])
        print(curr_data_batch)

    # shutdown everything to avoid zombies
    # sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(threads)




        #set placeholder




    # prelogits, _ = models.inception_v1.inference(image_batch, args.keep_probability,
    #                                  phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
    #                                  weight_decay=args.weight_decay)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_base_dir',type=str,help="Directory where to write events log",default='~/logs/template')
    parser.add_argument('--data_dir', type=str, help="Directory where to contain images",
                        default='~/logs/template')

    parser.add_argument('--max_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')

    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

if __name__ =='__main__':
    main(parse_arguments(sys.argv[1:]))