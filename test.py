import os
import tensorflow as tf
#create test data

def dump_numbers_to_file(fname, start_num, end_num):
  with open(fname, 'w') as f:
    for i in range(start_num, end_num):
      f.write(str(i)+"\n")



# num_files=10
# num_entries_per_file=10
# file_root="/home/tcl-admin/temp/pipeline"
# os.system('mkdir -p '+file_root)
# for fi in range(num_files):
#   fname = file_root+"/"+str(fi)
#   dump_numbers_to_file(fname, fi*num_entries_per_file, (fi+1)*num_entries_per_file)

def create_session():
    """Resets local session, returns new InteractiveSession"""
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3  # don't hog all vRAM
    config.operation_timeout_in_ms = 15000  # terminate on long hangs
    sess = tf.InteractiveSession("", config=config)
    return sess


filename_queue = tf.train.string_input_producer(["/temp/pipeline/0",
                                                 "/temp/pipeline/1"],
                                                shuffle=False)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
numeric_val1, = tf.decode_csv(value, record_defaults=[[-1]])
numeric_val2, = tf.decode_csv(value, record_defaults=[[-1]])
numeric_batch = tf.train.batch_join([numeric_val1,numeric_val2], 2)
# have to create session before queue runners because they use default session
sess = create_session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

print '\n'.join([t.name for t in threads])
for i in range(20):
  print sess.run([numeric_batch])

coord.request_stop()
coord.join(threads)