"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from PIL import Image
from models.BasicLSTM import configuration
from models.BasicLSTM.utils import inference_wrapper
from models.BasicLSTM.utils import caption_generator
from utils import vocabulary
from models.BasicLSTM.utils import inputs as input_ops

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "models/BasicLSTM/train_log",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "data/raw/word_counts_all.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_file_pattern", "data/tfrecord/test-tfrecord-00001-of-????",
                       "file path containing the tfrecord")


tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    # Build the inference graph.
    config = configuration.ModelConfig()
    config.input_file_pattern = FLAGS.input_file_pattern

    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(config, FLAGS.checkpoint_path)
        complete_sentences = input_ops.batch_input_data(
            file_name_pattern=config.input_file_pattern,
            config=config,
            mode='test'
        )

        # for the num_epoch parameter to be set
        init_local = tf.local_variables_initializer()
        init_global = tf.global_variables_initializer()
    g.finalize()


    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    tf.logging.info("Running Hypothesis & Reference generation.")

    # modest mode, not occupying all GPUs.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(graph=g, config=sess_config) as sess:
        # initialize all the variables especially for num_epochs
        sess.run(init_local)
        sess.run(init_global)
        # Load the model from checkpoint.
        restore_fn(sess)
        # start the queue
        threads = tf.train.start_queue_runners(sess=sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab, beam_size=5)

        # try:
        with open("models/BasicLSTM/test_log/hypo.txt", "w") as hf, open("models/BasicLSTM/test_log/ref.txt", "w") as rf:
            counter = 0
            while True:
                sentence_fetch = sess.run(complete_sentences)
                print("---------------------------------------")
                print("Test case No.{0}".format(counter))
                print("REF:\t", " ".join([x for x in map(vocab.id_to_word, sentence_fetch)]))
                # time.sleep(3)
                rf.write(" ".join([x for x in map(vocab.id_to_word, sentence_fetch)]) + '\n')

                captions = generator.beam_search(sess, sentence_fetch, k=1)

                caption = captions[0]
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence]
                sentence = " ".join(sentence)
                print("HYP:\t", sentence)
                hf.write(sentence + '\n')
                counter += 1

        # except:
        #     print("Done generating the captions.")


if __name__ == "__main__":
    tf.app.run()
