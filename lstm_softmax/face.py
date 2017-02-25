import tensorflow as tf
import numpy as np
import csv
import sys
import socket
import time
from contextlib import closing
from util.datareader import data_iterator_lstm as data_iterator

flags=tf.flags
logging=tf.logging
flags.DEFINE_string("train_data", None, "train_data")
flags.DEFINE_string("test_data", None, "test_data")
flags.DEFINE_integer("part", None, "part")
flags.DEFINE_bool("train", False, "train")
flags.DEFINE_bool("test", False, "test")
flags.DEFINE_bool("DemoServer", False, "demoServer")
flags.DEFINE_bool("InterServer", False, "InterServer")

FLAGS=flags.FLAGS

class Config(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_lstm_layers=2
    unroll_step=25
    sen_vec_size=800
    max_epoch = 6
    max_max_epoch = 39
    keep_prob=0.5
    lr_decay = 0.8
    batch_size=5
    face_vec_size=1
    class_size=10

def get_data_size(path):
    f=open(path,"rb")
    d=csv.reader(f)
    raw_data=[]
    for row in d:
        raw_data.append(row)
    return len(raw_data)

class FaceModel(object):
    def __init__(self, is_training=True, config=Config()):
        self._config=config
        self.unroll_step=unroll_step=config.unroll_step
        self.batch_size=batch_size=config.batch_size
        self.sen_vec_size=sen_vec_size=config.sen_vec_size
        self.face_vec_size=face_vec_size=config.face_vec_size
        self._x = tf.placeholder(tf.float32, shape=[batch_size, unroll_step, sen_vec_size], name="model_x")
        self._y_ = tf.placeholder(tf.int32, shape=[batch_size, unroll_step], name="model_y")
        #lstm cell based on: http://arxiv.org/pdf/1409.2329v5.pdf. which main contribution is the dropout regularization
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(sen_vec_size, forget_bias=1.0)
        #drop out output
        if is_training and config.keep_prob < 1:
            lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_lstm_layers)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        #drop out input
        inputs=self._x
        if is_training and config.keep_prob < 1:
            inputs=tf.nn.dropout(inputs, config.keep_prob)

        inputs=[tf.squeeze(input_, [1]) for input_ in tf.split(1, unroll_step, inputs)]
        outputs, state=tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

        self._final_state=state
        output = tf.reshape(tf.concat(1, outputs), [-1, sen_vec_size])
        softmax_w = tf.get_variable("softmax_w", [sen_vec_size, config.class_size])
        softmax_b = tf.get_variable("softmax_b", [config.class_size])
        y=tf.matmul(output, softmax_w) + softmax_b
        self._y = tf.nn.softmax(y)

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [y],
            [tf.reshape(self._y_, [-1])],
            [tf.ones([batch_size * unroll_step])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state 

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
#clip the gradient to fight for the exploding/vanishing problem, reference: http://arxiv.org/abs/1211.5063
#t_list[i] * (clip_norm / max(global_norm, clip_norm))
#tf.gradients returns the sum(dy/dx) for each x
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
#apply the clipped gradients
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._x

    @property
    def targets(self):
        return self._y_

    @property
    def config(self):
        return self._config

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def y(self):
        return self._y

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

def run_epoch(session, m, data, eval_op, verbose=False):
    epoch_size=get_data_size(data)
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    #for step in range(epoch_size):
    for step, (x, y) in enumerate(data_iterator(data, m.config.sen_vec_size, m.config.batch_size, m.config.unroll_step, FLAGS.part)):

        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.unroll_step

#use halfed l2 norm loss
        if verbose and step % (epoch_size // 10) == 10:
          print("%.3f perlexity: %.3f speed: %.0f sps" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

def run_epoch_test(session, m, data, eval_op, verbose=False):
    epoch_size=get_data_size(data)
    start_time = time.time()
    xs = []
    ys = []
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, t) in enumerate(data_iterator(data, m.config.sen_vec_size, m.config.batch_size, m.config.unroll_step, FLAGS.part)):

        y, cost, state, _ = session.run([m.y, m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: t,
                                      m.initial_state: state})
        ys.append(y)
        xs.append(x)
        
        costs += cost
        iters += m.unroll_step

    return np.exp(costs / iters), xs, ys



def train():
    if not FLAGS.train_data:
        raise ValueError("Must set --train_data")
    if not FLAGS.test_data:
        raise ValueError("Must set --test_data")
    config = Config()
    #dataset=DataSet(FLAGS.train_data, config)
    #train_data = valid_data = dataset
    train_data = valid_data = FLAGS.train_data

    eval_config = Config()
    eval_config.batch_size = 1
    eval_config.unroll_step = 1
    
    #test_data=DataSet(FLAGS.test_data, eval_config)
    test_data=FLAGS.test_data
    
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = FaceModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = FaceModel(is_training=False, config=config)
            mtest = FaceModel(is_training=False, config=eval_config)
  
        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

            train_perlexity = run_epoch(session, m, train_data, m.train_op, verbose=True)
            print("Epoch: %d Train perlexity: %.3f" % (i + 1, train_perlexity))

            valid_perlexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid perlexity: %.3f" % (i + 1, valid_perlexity))

        test_perlexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test perlexity: %.3f" % test_perlexity)
        saver = tf.train.Saver()
        save_path = saver.save(session, "faceModel"+str(FLAGS.part)+".ckpt")
        print("Model saved in file: %s" % save_path)

def serialYS(ys):
    msg = ""
    for y in ys: 
        y=y.tolist()
        for i in y:
            i=str(i).replace(" ","")[1:-1]
            msg+=str(i)
        msg+="\n"
    return msg
    

def test():
    print "test"
    eval_config = Config()
    eval_config.batch_size = 1
    eval_config.unroll_step = 1
    #test_data=DataSet(FLAGS.test_data, eval_config)
    test_data=FLAGS.test_data
    with tf.Session() as session:
        initializer = tf.random_uniform_initializer(-eval_config.init_scale,
                                                    eval_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = FaceModel(is_training=False, config=eval_config)
        saver = tf.train.Saver()
        saver.restore(session, "faceModel"+str(FLAGS.part)+".ckpt")
        print("Model restored.")
        test_perlexity, _, ys = run_epoch_test(session, mtest, test_data, tf.no_op())
        print("Test perlexity: %.3f" % test_perlexity)
        #print("len xs: %d" % len(xs))
        print("len ys: %d" % len(ys))
        #msg=serialYS(xs)
        #f=open("x", "w+")
        #f.write(msg)
        #f.close()
        msg=serialYS(ys)
        f=open("y"+str(FLAGS.part)+".data", "w+")
        f.write(msg)
        f.close()

def demoServer():
    print "Run server"
    serversocket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM) 
    port = 9999                  
    serversocket.bind(("", port))
    serversocket.listen(5)       

    eval_config = Config()
    eval_config.batch_size = 1
    eval_config.unroll_step = 1
    #test_data=DataSet(FLAGS.test_data, eval_config)
    test_data=FLAGS.test_data

    session=tf.Session()

    initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtest = FaceModel(is_training=False, config=eval_config)
    saver = tf.train.Saver()
    saver.restore(session, "faceModel"+str(FLAGS.part)+".ckpt")
    print("Model restored.")
 

    with closing(serversocket):
        while True:
            print "Listening"
            clientsocket,addr = serversocket.accept()      
            print("Got a connection from %s" % str(addr))
            #just send all the results
            #get test results
            with closing(clientsocket):
                    with session.as_default():
                        test_perlexity, _, ys = run_epoch_test(session, mtest, test_data, tf.no_op())
                        print("Test perlexity: %.3f" % test_perlexity)
                        msg=serialYS(ys)
                        clientsocket.send(msg)

    session.close()
    return

def interServer():
    pass

def main(_):
    if not FLAGS.part:
        print("no --part, going to calculate everything, the computer may break down")
    if FLAGS.train:
        train()
    elif FLAGS.DemoServer:
        demoServer()
    elif FLAGS.InterServer:
        interServer()
    elif FLAGS.test:
        test()
    else:
        raise ValueError("Must set --train/--test/--DemoServer/--InterServer")

if __name__=="__main__":
    tf.app.run()
