# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

import tensorflow as tf
import reader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None, "Model output directory.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class LSTMInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.bidirectional_producer(
                data, batch_size, num_steps, name=name)


class LSTMModel(object):
    """The LSTM Model."""
    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        print ("vocab size: ", vocab_size)
        print ("size: ", size)
        print ("batch size:", batch_size)
        

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        #print("output: ", output)

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._proba = tf.nn.softmax(logits)
        
        #print("PROBA: ", self._proba)
        #print("PROBA shape", self._proba.get_shape())
        #indices = [[0], [1]]
        #print("values: ", tf.gather_nd(self._proba, indices))
        #print("INDEX:", list(self._proba.eval(session=se[0]).index(max(self._proba.eval(session=sess)[0])))
        #print('logits shape:', logits.shape()) 
        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def proba(self):
        return self._proba
        
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 60000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 60000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 60000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 1 #2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1 #20
    vocab_size = 60000


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            "proba": model.proba, # added to test proba
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        #print("initial state:", model.initial_state)
        for i, (c, h) in enumerate(model.initial_state):
            #print("c:", c, "h: ", h)
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
	# prob is n x V where n is number of sentences, V is size of vocab
        proba = vals["proba"] # added to test proba
        
        #print ("proba", proba)
        #print ("proba size:", np.shape(proba))
        #print ("state", state)
        #print ("state size: ", np.shape(state))
        #print(list(proba[0]).index(max(proba[0])))
         
        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                         iters * model.input.batch_size / (time.time() - start_time)))


    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    word_to_id = reader._build_vocab('dataset/treebank2/raw/wsj/')
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to LSTM data directory")

    raw_data = reader._raw_data(FLAGS.data_path)
    train_data, train_sentences, test_data, test_sentences, _ = raw_data
    #print("data, sentences")
    #print(train_data[:10], train_sentences[:10], test_data[:10], test_sentences[:10])
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

        with tf.name_scope("Train"):
            train_input = LSTMInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = LSTMModel(is_training=True, config=config, input_=train_input)
            # tf.contrib.deprecated.scalar_summary("Training Loss", m.cost)
            # tf.contrib.deprecated.scalar_summary("Learning Rate", m.lr)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        # with tf.name_scope("Valid"):
        #     valid_input = LSTMInput(config=config, data=valid_data, name="ValidInput")
        #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #         mvalid = LSTMModel(is_training=False, config=config, input_=valid_input)
        #     tf.contrib.deprecated.scalar_summary("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            print("test_data:", test_sentences)
            test_data_words = test_sentences[0].split()
            choices = test_data_words[:5] # get five choices
            #new_test_data = ''.join(test_data_words[5:])
            test_input = LSTMInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = LSTMModel(is_training=False, config=eval_config,
                                                 input_=test_input)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                #max_word_index = list(proba[0]).index(max(proba[0]))
                print('choices:', choices)
                print('word_to_id choices:', word_to_id[choices[0]])
                i1, i2, i3, i4, i5 = word_to_id[choices[0]], word_to_id[choices[1]], word_to_id[choices[2]], word_to_id[choices[3]], word_to_id[choices[4]]
                prob1 = session.run(m.proba)[0][i1]
                prob2 = session.run(m.proba)[0][i2]
                prob3 = session.run(m.proba)[0][i3]
                prob4 = session.run(m.proba)[0][i4]
                prob5 = session.run(m.proba)[0][i5]
                
                # get the answer choice that was most likely
                word_to_prob = [(choices[0], prob1), (choices[1], prob2), (choices[2], prob3), (choices[3], prob4), (choices[4], prob5)]
                print("word to prob:", word_to_prob)
                max_word = max(word_to_prob, key=lambda x: x[1])[0]
                print("word!!", max_word)
                
                result = session.run(m.proba) #result = session.run(tf.gather_nd(m.proba, indices))
                print("result: ", result)

                # valid_perplexity = run_epoch(session, mvalid)
                # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            # Save model to file
            FLAGS.save_path = './saved-models/LSTM-forward-model'
            print("Saving model to %s." % FLAGS.save_path)
            saver = tf.train.Saver()
            session.run(tf.initialize_all_variables())
            saver.save(session, FLAGS.save_path)

            # Load model from save file
            saved_path = FLAGS.save_path + '.meta'
            new_saver = tf.train.import_meta_graph(saved_path)
            new_saver.restore(session, FLAGS.save_path)

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
        tf.app.run()

