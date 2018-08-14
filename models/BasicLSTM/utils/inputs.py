from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

from models.BasicLSTM.model import AutoEncoder


def _parse_sequence_example(example):
    """
    :param example:
    :return: sentence
    """
    _, features = tf.parse_single_sequence_example(
        example,
        sequence_features={"sentence/sentence": tf.FixedLenSequenceFeature([], dtype=tf.int64)}
    )

    return features["sentence/sentence"]


def batch_input_data(file_name_pattern, config, mode='train'):
    """
    Read data from the tfrecord and return batch.
    :param file_name_pattern: a list of the file name pattern indicating the class of the file.
    :param config:  a configuration object
    :param shard_queue_name:    name of the file name queue
    :param mode: mode of current session
    :return: batched images: left & right,
    """
    assert mode in ['train', 'test', 'val']

    files = tf.gfile.Glob(file_name_pattern)
    if not files:
        tf.logging.fatal("No TFRecord file found.")

    reader = tf.TFRecordReader()

    # input the sentence once or repetitvely
    if mode == 'train':
        filename_queue = tf.train.string_input_producer(files, shuffle=True)
    elif mode == 'test':
        filename_queue = tf.train.string_input_producer(files, shuffle=True, num_epochs=1)
    else:
        filename_queue = tf.train.string_input_producer(files, shuffle=True, num_epochs=1)

    _, example = reader.read(filename_queue)

    sentence = _parse_sequence_example(example)
    if mode == 'test':
        return sentence

    caption_length = tf.shape(sentence)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

    input_seq = tf.slice(sentence, [0], input_length)
    target_seq = tf.slice(sentence, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)


    complete_sentences, input_seqs, target_seqs, masks = tf.train.batch(
        [sentence, input_seq, target_seq, indicator],
        batch_size=config.batch_size,
        capacity=config.batch_size*2,
        dynamic_pad=True,
        name="batch_and_pad"
    )

    return complete_sentences, input_seqs, target_seqs, masks



