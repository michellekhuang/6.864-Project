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
        self.vocab_size = config.vocab_size

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
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
    num_steps = 1
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 60000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 1
    vocab_size = 60000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 1
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 1
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
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        proba = vals["proba"] # added to test proba
        if model.input.epoch_size != 0:
            if verbose and step % (model.input.epoch_size // 25) == 10:
                print("Progress is %.3f, data size is %.3f" % (step * 1.0 / model.input.epoch_size, model.input.epoch_size))


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

# given the sentence (in words) and sentence (in numbers) where the first 5 words are answer options abc
# finds the best answer choice from the model
# returns best answer as tuple (letter, word, word_id, probability)
def find_answer_probs(session, model, answer_words, word_to_id):
    answers = []
    choices = 'abcde'
    for i in range(5):
        prob = 0
        letter = choices[i]
        word_id = 0
        if answer_words[i] in word_to_id:
            word_id = word_to_id[answer_words[i]]
            prob = session.run(model.proba)[0][word_id]
        answers.append((letter, answer_words[i], word_id, prob))
    return answers
        
def create_test_tensor(sentence_test_data, eval_config, initializer):
    with tf.name_scope("Test"):
        test_input = LSTMInput(config=eval_config, data=sentence_test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            test_model = LSTMModel(is_training=False, config=eval_config,
                                             input_=test_input)
    return test_model

def main(_):
    word_to_id = reader._build_vocab('dataset/treebank2/raw/wsj/')
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to LSTM data directory")

    raw_data = reader._raw_data(FLAGS.data_path)
    word_to_id, train_data, test_sentences, test_data_in_list_of_lists, question, answer = raw_data
    config = get_config()
    eval_config = get_config()

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
        
        mtests = []
        for i in range(len(test_sentences)):
            print('Creating test tensor', i)
            # can't run through RNN because not enough values
            if len(test_data_in_list_of_lists[i]) < 2:
                mtests.append(None)
            else:
                new_test = create_test_tensor(test_data_in_list_of_lists[i], eval_config, initializer)
                mtests.append(new_test)
            
        # with tf.name_scope("Valid"):
        #     valid_input = LSTMInput(config=config, data=valid_data, name="ValidInput")
        #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #         mvalid = LSTMModel(is_training=False, config=config, input_=valid_input)
        #     tf.contrib.deprecated.scalar_summary("Validation Loss", mvalid.cost)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)

        with sv.managed_session() as session:

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                run_epoch(session, m, eval_op=m.train_op, verbose=True)
            
            with open('forward_out.txt', 'w') as f:
                num_correct = 0.0
                for i in range(len(test_sentences)):
                    mtest = mtests[i]
                    if mtest is None:
                        print('Not enough info skipping question', i)
                        continue
                    run_epoch(session, mtest)
                    choices = find_answer_probs(session, mtest, test_sentences[i].split()[0:5], word_to_id)
                    
                    f.write(str(i) + ' ')
                    
                    for choice in choices:
                        f.write(str(choice[3]) + ' ')
                    
                    f.write('\n')
                        
                    best_answer = max(choices, key=lambda x: x[3])
                    print('Best answer')
                    print(best_answer)
                    print('Correct answer')
                    print('Answer to problem', i+1, 'is', answer[str(i+1)])
                    if best_answer[0] == answer[str(i+1)]:
                        num_correct += 1
                        
                f.write('\n')
                
                f.write('Accuracy: ' + str(num_correct/len(test_sentences)))
                        
                print()
                print('Accuracy is', num_correct/len(test_sentences))


if __name__ == "__main__":
        tf.app.run()

