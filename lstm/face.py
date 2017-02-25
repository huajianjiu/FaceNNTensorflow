import tensorflow as tf
import numpy as np
import csv
import model.docModel as docModel
import sys
import socket
import time
from contextlib import closing

flags=tf.flags
logging=tf.logging
flags.DEFINE_string("train_data", None, "train_data")
flags.DEFINE_string("test_data", None, "test_data")
flags.DEFINE_bool("train", False, "train")
flags.DEFINE_bool("test", False, "test")
flags.DEFINE_bool("DemoServer", False, "demoServer")
flags.DEFINE_bool("InterServer", False, "InterServer")

FLAGS=flags.FLAGS

class Config100(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_lstm_layers=2
    unroll_step=20
    sen_vec_size=100
    max_epoch = 4
    max_max_epoch = 13
    keep_prob=1.0
    lr_decay = 0.5
    batch_size=20
    face_vec_size=12

class Config(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_lstm_layers=2
    unroll_step=20
    sen_vec_size=800
    max_epoch = 6
    max_max_epoch = 39
    keep_prob=1.0
    lr_decay = 0.8
    batch_size=2
    face_vec_size=12



#ones/zeros make things worse
def get_data_size(path):
    f=open(path,"rb")
    dataReader=csv.reader(f)
    raw_data=[]
    for row in dataReader:
        raw_data.append(row)
    return len(raw_data)

def data_iterator(path, config):
    sen_vec_size=config.sen_vec_size
    face_vec_size=config.face_vec_size
    batch_size=config.batch_size
    num_steps=config.unroll_step
    f=open(path,"rb")
    dataReader=csv.reader(f)
    raw_data=[]
    for row in dataReader:
        raw_data.append(row)
    data_len=len(raw_data)
    raw_data_x=np.zeros([data_len, sen_vec_size], dtype=np.float32)
    raw_data_y=np.zeros([data_len, face_vec_size], dtype=np.float32)
    for i in range(data_len):
        raw_data_x[i]=np.array(docModel.getVector(raw_data[i][0]), dtype=np.float32)
        #remove the 5-8th, 11th, 12th, 19th parameters because they is almost not used.
        raw_data_y[i]=np.array(raw_data[i][1:5]+raw_data[i][9:11]+raw_data[i][13:19], dtype=np.float32)
        #use only a paramter
        #raw_data_y[i]=np.array(raw_data[i][17], dtype=np.float32)
    batch_len=data_len//batch_size
    data_x=np.zeros([batch_size, batch_len, sen_vec_size], dtype=np.float32)
    data_y=np.zeros([batch_size, batch_len, face_vec_size], dtype=np.float32)
    for i in range(batch_size):
        data_x[i] = raw_data_x[batch_len * i:batch_len * (i + 1)]
        data_y[i] = raw_data_y[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data_x[:, i*num_steps:(i+1)*num_steps]
        y = data_y[:, i*num_steps:(i+1)*num_steps]
        yield (x, y)

class FaceModel(object):
    #put all together, easy to be save
    def __init__(self, is_training=True, config=Config()):
        #Size of the Sentence Vector = 100
        self._config=config
        self.unroll_step=unroll_step=config.unroll_step
        self.batch_size=batch_size=config.batch_size
        self.sen_vec_size=sen_vec_size=config.sen_vec_size
        self.face_vec_size=face_vec_size=config.face_vec_size
        self._x = tf.placeholder(tf.float32, shape=[batch_size, unroll_step, sen_vec_size], name="model_x")
        self._y_ = tf.placeholder(tf.float32, shape=[batch_size, unroll_step, face_vec_size], name="model_y")
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
        relu_w = tf.get_variable("relu_w", [sen_vec_size, face_vec_size])
        relu_b = tf.get_variable("relu_b", [face_vec_size])
        #Use ReLU
        #y=tf.nn.relu(tf.matmul(output, relu_w) + relu_b)
        #do not use ReLU
        y=tf.matmul(output, relu_w) + relu_b
        self._y = y

#cross enrtrpy is not for this task. use root mean squared error / or mean squeared error / or sum squared error (=l2 norm loss)
#        loss = tf.nn.seq2seq.sequence_loss_by_example(
#            [y],
#            [tf.reshape(self._y_, [-1])],
#            [tf.ones([batch_size * unroll_step])])
#use mse
#        loss = sequence_mse_loss([y],[tf.reshape(self._y_, [-1, face_vec_size])])
#it seems ok to just use HALF squared error (tf.nn.l2_loss) the nn.l2_loss computes HALF the L2 norm
        t=tf.reshape(self._y_, [-1, face_vec_size])
        loss = tf.nn.l2_loss(y-t)
#get mean squared error?
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
#clip the gradient to fight for the exploding/vanishing problem, reference: http://arxiv.org/abs/1211.5063
#even relu layer do not need this, the lstm layers needs(lstm units use tanh) TODO: check if it brings problems to the relu layer
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
    for step, (x, y) in enumerate(data_iterator(data, m.config)):

        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.unroll_step

#use halfed l2 norm loss
        if verbose and step % (epoch_size // 10) == 10:
          print("%.3f mse: %.3f speed: %.0f sps" %
                (step * 1.0 / epoch_size, (2 * costs / iters),
                 iters * m.batch_size / (time.time() - start_time)))

    return (2 * costs / iters)

def run_epoch_test(session, m, data, eval_op, verbose=False):
    epoch_size=get_data_size(data)
    start_time = time.time()
    xs = []
    ys = []
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, t) in enumerate(data_iterator(data, m.config)):

        y, cost, state, _ = session.run([m.y, m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: t,
                                      m.initial_state: state})
        ys.append(y)
        xs.append(x)
        
        costs += cost
        iters += m.unroll_step

    return (2 * costs / iters), xs, ys



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

            train_mse = run_epoch(session, m, train_data, m.train_op, verbose=True)
            print("Epoch: %d Train MSE: %.3f" % (i + 1, train_mse))

            valid_mse = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid MSE: %.3f" % (i + 1, valid_mse))

        test_mse = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test MSE: %.3f" % test_mse)
        saver = tf.train.Saver()
        save_path = saver.save(session, "faceModel.ckpt")
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
        saver.restore(session, "faceModel.ckpt")
        print("Model restored.")
        test_mse, _, ys = run_epoch_test(session, mtest, test_data, tf.no_op())
        print("Test MSE: %.3f" % test_mse)
        #print("len xs: %d" % len(xs))
        print("len ys: %d" % len(ys))
        #msg=serialYS(xs)
        #f=open("x", "w+")
        #f.write(msg)
        #f.close()
        msg=serialYS(ys)
        f=open("y.data", "w+")
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
    saver.restore(session, "faceModel.ckpt")
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
                        test_mse, _, ys = run_epoch_test(session, mtest, test_data, tf.no_op())
                        print("Test MSE: %.3f" % test_mse)
                        msg=serialYS(ys)
                        f=open("y", "w+")
                        f.write(msg)
                        f.close()
                        clientsocket.send(msg)

    session.close()
    return

def interServer():
    pass

def main(_):
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
