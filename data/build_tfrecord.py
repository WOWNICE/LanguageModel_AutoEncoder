import tensorflow as tf
import os
import random
from utils.vocabulary import Vocabulary
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from data.name_processing import frequent_name_extract

# constant definition
word_file_path = "data/raw/word_counts_all.txt"
tfrecord_path = "data/tfrecord"
data_file_path = "data/raw/Gutenberg/*.txt"
name_file_path = "data/raw/names/*.txt"

def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _to_sequence_example(sentence, vocab, name_dict):
    """
    transfer the sentence into a sequence of ids
    :param sentence: a list of the words
    :return: sequence_example
    """

    # sentence -> name replacement -> lower case -> unk token -> start & end token
    sentence = [name_dict[_] if _ in name_dict else _ for  _ in sentence]
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


def record_write(sentences, num_tfrecord, prefix, output_file_path, vocab, name_dict):
    """
    write the record into the file path
    :param sentences:
    :param num_tfrecord:
    :param prefix:
    :param output_file_path:
    :param vocab:
    :return:
    """
    # split function from stackoverflow
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    # training set writing
    file_shards = list(split(sentences, num_tfrecord))

    print("Writing training set...")
    for _ in range(num_tfrecord):
        # create writer tfrecord
        # name = "tfrecord-00???-of00???"
        name = prefix+ '-tfrecord-' + (5-len(str(_)))*'0' + str(_) + '-of-00' + str(num_tfrecord)
        writer = tf.python_io.TFRecordWriter(os.path.join(output_file_path, name))

        for sentence in file_shards[_]:
            example = _to_sequence_example(sentence=sentence, vocab=vocab, name_dict=name_dict)
            if example is not None:
                writer.write(example.SerializeToString())

        print("Wrote {0} sentences in Record {2}, No.{1}".format(len(file_shards[_]), _, prefix))
        writer.close()


def build_dataset(data_file_pattern,
                  output_file_path,
                  vocab_file_path,
                  min_length=10,
                  max_length=60,
                  num_tfrecord_train=256,
                  num_tfrecord_val=16,
                  num_tfrecord_test=16):
    # build the vocabulary object
    print("Building vocabulary...")
    vocab = Vocabulary(vocab_file=vocab_file_path)

    # build the name dict
    print("Building name dictionary...")
    name_dict = frequent_name_extract(name_file_path=name_file_path)

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
    for i in range(len(txt_files)):
        with open(txt_files[i], "r") as f:
            try:
                content = f.read()
            except:
                continue
        sentences = sent_tokenize(content)
        sentences = [_ for _ in map(word_tokenize, sentences)]
        sentences = [_ for _ in filter(lambda x: len(x) <= max_length and len(x) >= min_length, sentences)]
        all_sentences.extend(sentences)
        print("Done Processing No.{0} books.".format(i))

    print("Shuffling sentences...")
    random.shuffle(all_sentences)
    print("Total sentences: " + str(len(all_sentences)))

    train_sentences = all_sentences[:int(0.8*len(all_sentences))]
    val_sentences = all_sentences[int(0.8 * len(all_sentences)):int(0.9*len(all_sentences))]
    test_sentences = all_sentences[int(0.9*len(all_sentences)):]

    if not tf.gfile.Exists(output_file_path):
        print("No existing output file path, built it.")
        tf.gfile.MakeDirs(output_file_path)

    # write training set / validation set / test set.
    record_write(train_sentences, num_tfrecord=num_tfrecord_train, prefix='train', output_file_path=output_file_path, vocab=vocab, name_dict=name_dict)
    record_write(val_sentences, num_tfrecord=num_tfrecord_val, prefix='val', output_file_path=output_file_path, vocab=vocab, name_dict=name_dict)
    record_write(test_sentences, num_tfrecord=num_tfrecord_test, prefix='test', output_file_path=output_file_path, vocab=vocab, name_dict=name_dict)


if __name__ == "__main__":
    build_dataset(
        # data_file_pattern="data/raw/Gutenberg_test/*.txt",
        data_file_pattern=data_file_path,
        output_file_path=tfrecord_path,
        vocab_file_path=word_file_path,
        min_length=10,
        max_length=60,
        num_tfrecord_train=256,
        num_tfrecord_val=16,
        num_tfrecord_test=16
    )