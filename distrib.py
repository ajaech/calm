import bunch
import json
import tensorflow as tf

from vocab import Vocab
from batcher import ReadData, Dataset
from model import HyperModel

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

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    with open('default_params.json', 'r') as f:
      params = bunch.Bunch(json.load(f))
    usernames, texts = ReadData('/n/falcon/s0/ajaech/clean.tsv.bz', limit=200000, mode='train')
    dataset = Dataset(max_len=params.max_len + 1, preshuffle=True)
    dataset.AddDataSource(usernames, texts)

    vocab = Vocab.MakeFromData(texts, min_count=20)
    username_vocab = Vocab.MakeFromData([[u] for u in usernames],
                                        min_count=50)
    dataset.Prepare(vocab, username_vocab)


    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      global_step = tf.Variable(0)
      model = HyperModel(params, len(vocab), len(username_vocab), use_nce_loss=True)

      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
      optimizer = tf.train.AdamOptimizer(0.001)
      train_op = optimizer.apply_gradients(zip(grads, tvars))

      saver = tf.train.Saver()
      # init_op = tf.global_variables_initializer()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/n/falcon/s0/ajaech/tmp",
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
        
        cost, _ = sess.run([model.cost, train_op], feed_dict=feed_dict)

        print float(cost)

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
