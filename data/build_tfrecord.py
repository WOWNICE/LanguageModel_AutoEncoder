import tensorflow as tf
import os
import random
from utils.vocabulary import Vocabulary
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# constant definition
word_file_path = "data/raw/word_counts_all.txt"
tfrecord_path = "data/tfrecord"
data_file_path = "data/raw/Gutenberg/*.txt"

def _int64_feature(value):
  """ 封装函数：插入int64 Feature到SequenceExample proto
  Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
  """封装函数：插入int64 FeatureList到SequenceExample proto
  Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _to_sequence_example(sentence, vocab):
    """
    transfer the sentence into a sequence of ids
    :param sentence: a list of the words
    :return: sequence_example
    """

    # sentence -> lower case -> unk token -> start & end token
    sentence = [_ for _ in map(lambda x: x.lower(), sentence)]
    word_ids = [vocab.start_id] + [_ for _ in map(lambda x: vocab.word_to_id(x), sentence)] + [vocab.end_id]

    # check whether unk situation is severe or not
    # count = 0
    # for word in word_ids:
    #     if word == vocab.unk_id:
    #         count += 1
    # if count > 5:
    #     print(count, len(word_ids) - 2)

    feature_lists = tf.train.FeatureLists(feature_list={
        "sentence/sentence": _int64_feature_list(word_ids)
    })

    sequence_example = tf.train.SequenceExample(feature_lists=feature_lists)

    return sequence_example


def build_dataset(data_file_pattern, output_file_path, vocab_file_path, num_tfrecord=256):
    # build the vocabulary object
    print("Building vocabulary...")
    vocab = Vocabulary(vocab_file=vocab_file_path)

    # 1. build all the txt in one list
    # 2. use nltk package to tockenize all the sentence
    # 3. shuffle the sentences
    # 4. save to the tfrecord file


    txt_files = tf.gfile.Glob(data_file_pattern)
    if not txt_files:
        print("No images found in given path.")
        return

    print("Processing sentences...")
    all_sentences = []
    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            content = f.read()
        sentences = sent_tokenize(content)
        sentences = [_ for _ in map(word_tokenize, sentences)]
        all_sentences.extend(sentences)

    # random.shuffle(all_sentences)
    print("Total sentences: " + str(len(all_sentences)))

    if not tf.gfile.Exists(output_file_path):
        print("No existing output file path, built it.")
        tf.gfile.MakeDirs(output_file_path)

    # split function from stackoverflow
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    file_shards = list(split(all_sentences, num_tfrecord))

    print("Writing tfrecords...")
    for _ in range(num_tfrecord):
        # create writer tfrecord
        writer = tf.python_io.TFRecordWriter(os.path.join(output_file_path, 'tfrecord-' + str(_) + '-of-' + str(num_tfrecord)))

        for sentence in file_shards[_]:
            example = _to_sequence_example(sentence=sentence, vocab=vocab)
            if example is not None:
                writer.write(example.SerializeToString())

        print("Wrote {0} sentences in Record No.{1}".format(len(file_shards[_]), _))
        writer.close()

if __name__ == "__main__":
    build_dataset(
        data_file_pattern=data_file_path,
        output_file_path=tfrecord_path,
        vocab_file_path=word_file_path,
        num_tfrecord=128
    )