# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.BasicLSTM import inputs as input_ops


class AutoEncoder(object):
    def __init__(self, config, mode):
        """Basic setup.

        Args:
          config: Object containing configuration parameters.
          mode: "train", "eval" or "inference".
          train_inception: Whether the inception submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode

        self.reader = tf.TFRecordReader()

        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        """Returns true if the model is built for training mode.
        """
        return self.mode == "train"

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
          self.image_streams
          self.input_seqs
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)
        """
        if self.mode == "inference":
            # In inference mode, inputs are fed via placeholders.
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[None],  # batch_size
                                        name="input_feed")

            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None
        else:
            input_seqs, target_seqs, input_mask = input_ops.batch_input_data(
                file_name_pattern=self.config.input_file_pattern,
                config=self.config
            )

        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_seq_embeddings(self):
        """Builds the input sequence embeddings.
        生caption中的word序列的embedding特征

        Inputs:
          self.input_seqs

        Outputs:
          self.seq_embeddings
        """
        # cpu上执行word序列的embedding特征(矩阵查询方式)
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

    def build_model(self):
        """Builds the model.

        Inputs:
            self.seq_embeddings
            self.target_seqs (training and eval only)
            self.input_mask (training and eval only)

        Outputs:
            self.total_loss (training and eval only)
            self.target_cross_entropy_losses (training and eval only)
            self.target_cross_entropy_loss_weights (training and eval only)
        """

        with tf.variable_scope("lstm", initializer=self.initializer, reuse=tf.AUTO_REUSE) as lstm_scope:

            lstm_cell_enc = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.config.num_lstm_units, state_is_tuple=True, name="lstm_encoder")

            lstm_cell_dec = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.config.num_lstm_units, state_is_tuple=True, name="lstm_decoder")

            # drop out
            if self.mode == "train":
                lstm_cell_enc = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell_enc,
                    input_keep_prob=self.config.lstm_dropout_keep_prob,
                    output_keep_prob=self.config.lstm_dropout_keep_prob)
                lstm_cell_dec = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell_dec,
                    input_keep_prob=self.config.lstm_dropout_keep_prob,
                    output_keep_prob=self.config.lstm_dropout_keep_prob)

            # Feed the image embeddings to set the initial LSTM state.
            zero_state_enc = lstm_cell_enc.zero_state(
                batch_size=self.input_seqs.get_shape()[0], dtype=tf.float32)

            # sequence_length is for correctness
            _, enc_final_state = tf.nn.dynamic_rnn(cell=lstm_cell_enc,
                                                   inputs=self.seq_embeddings,
                                                   sequence_length=None,
                                                   initial_state=zero_state_enc,
                                                   dtype=tf.float32,
                                                   scope=lstm_scope)

            # lstm state has two states c & h,
            # and h is the final representation of the sentence,
            # which would be fed as multi-modal input.
            static_feature = enc_final_state.h

            # use zero state is the initial state while the intermediate representation serves as multi-modal input.
            initial_state_dec = lstm_cell_dec.zero_state(
                batch_size=self.input_seqs.get_shape()[0], dtype=tf.float32)

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()

            if self.mode == "inference":
                # In inference mode, use concatenated states for convenient feeding and fetching.
                tf.concat(axis=1, values=initial_state_dec, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell_dec.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                # Run a single LSTM step.
                lstm_outputs, state_tuple = lstm_cell_dec(
                    inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
                    state=state_tuple)

                # Concatentate the resulting state.
                tf.squeeze(tf.concat(axis=1, values=state_tuple), name="state")
            else:
                # Run the batch of sequence embeddings through the LSTM.
                # first construct the multi-modal input.
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                static_feature = tf.expand_dims(static_feature, axis=1)
                batch_time_static_feature = tf.tile(static_feature, [1, tf.reduce_max(sequence_length), 1])
                combined_inputs = tf.concat([batch_time_static_feature, self.seq_embeddings], axis=2)

                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell_dec,
                                                    inputs=combined_inputs,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state_dec,
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)

        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell_dec.output_size])

        # fully connected layer for classification
        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=self.config.vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)

        if self.mode == "inference":
            tf.squeeze(tf.nn.softmax(logits), name="softmax")
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            # compute the loss
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                    logits=logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                tf.reduce_sum(weights),
                                name="batch_loss")

            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()

            # add summary
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

    def setup_global_step(self):
        """Sets up the global step Tensor.
        """
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_global_step()
