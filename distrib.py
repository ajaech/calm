import bunch
import json
import logging
import os
import time

import tensorflow as tf

from vocab import Vocab
from batcher import ReadData, Dataset
from model import HyperModel

tf.app.flags.DEFINE_string("expdir", "",
                           "Where to save all the files")

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)


  with open('default_params.json', 'r') as f:
    params = bunch.Bunch(json.load(f))

  if FLAGS.job_name == "ps":

    if FLAGS.task_index == 0:
      if not os.path.exists(FLAGS.expdir):
        os.mkdir(FLAGS.expdir)
      else:
        print 'Error: expdir already exists'
        exit()

      usernames, texts = ReadData('/n/falcon/s0/ajaech/clean.tsv.bz', mode='train')
      dataset = Dataset(max_len=params.max_len + 1, preshuffle=True)
      dataset.AddDataSource(usernames, texts)
    
      vocab = Vocab.MakeFromData(texts, min_count=20)
      username_vocab = Vocab.MakeFromData([[u] for u in usernames],
                                        min_count=50)  

      vocab.Save(os.path.join(FLAGS.expdir, 'word_vocab.pickle'))
      username_vocab.Save(os.path.join(FLAGS.expdir, 'username_vocab.pickle'))

      os.mkdir(os.path.join(FLAGS.expdir, 'READY'))

    server.join()
   
  elif FLAGS.job_name == "worker":

    usernames, texts = ReadData(
      '/n/falcon/s0/ajaech/clean.tsv.bz',
      mode='train',
      worker=FLAGS.task_index,
      num_workers=len(worker_hosts)
    )
    dataset = Dataset(max_len=params.max_len + 1, preshuffle=True)
    dataset.AddDataSource(usernames, texts)

    while not os.path.exists(os.path.join(FLAGS.expdir, 'READY')):
      print('waiting for READY file')
      time.sleep(1)  # wait for ready file
    print 'READY'

    vocab = Vocab.Load(os.path.join(FLAGS.expdir, 'word_vocab.pickle'))
    username_vocab = Vocab.Load(os.path.join(FLAGS.expdir, 'username_vocab.pickle'))
    print 'preparing dataset'
    dataset.Prepare(vocab, username_vocab)

    print 'building model'
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      global_step = tf.Variable(0)
      model = HyperModel(params, len(vocab), len(username_vocab), use_nce_loss=True)

      optimizer = tf.train.AdagradOptimizer(0.01)
      """optimizer = tf.train.SyncReplicasOptimizer(
        optimizer,
        replicas_to_aggregate=3,
        total_num_replicas=len(worker_hosts),
        replica_id=FLAGS.task_index,
        name='syncer')"""
      train_op = optimizer.minimize(
        model.cost, global_step=global_step)

      saver = tf.train.Saver()
      # init_op = tf.global_variables_initializer()
      init_op = tf.initialize_all_variables()

    logging.basicConfig(
      filename=os.path.join(FLAGS.expdir, 'worker{0}.log'.format(FLAGS.task_index)),
      level=logging.INFO)

    # Create a "supervisor", which oversees the training process.
    print 'creating supervisor'
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir=FLAGS.expdir,
                             init_op=init_op,
                             summary_op=None,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      while not sv.should_stop() and step < 1000000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        s, seq_len, usernames = dataset.GetNextBatch()
        
        feed_dict = {
          model.x: s[:, :-1],
          model.y: s[:, 1:],
          model.seq_len: seq_len,
          model.username: usernames
        }        
        
        cost, _, step = sess.run([model.cost, train_op, global_step], 
                                 feed_dict=feed_dict)

        if step % 10 == 0:
          logging.info({'iter': step, 'cost': float(cost)})
          print step, cost

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
