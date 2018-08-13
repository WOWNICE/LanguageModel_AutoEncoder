"""
Module to test inputs building module.
"""

import tensorflow as tf
import numpy as np
from models.BasicLSTM.inputs import batch_input_data
from models.BasicLSTM.configuration import ModelConfig

file_pattern = "data/tfrecord/tfrecord-?-of-128"

def main():
    config = ModelConfig()
    input_seqs, target_seqs, masks = batch_input_data(file_pattern, config)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        for _ in range(2):
            print(40*"-")
            input_val = sess.run(input_seqs)
            targe_val = sess.run(target_seqs)
            mask_val = sess.run(masks)

            # print(left_images_val)
            # print(right_images_val)
            print(type(input_val))
            print(np.array(input_val).shape)

if __name__ == "__main__":
    main()