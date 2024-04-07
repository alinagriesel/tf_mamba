# imports
import sentencepiece as sp
import re
import tensorflow_text as tft
import tensorflow as tf

def data_cleaning():
    """
    """
    # read the text file
    bible_file = open("bible.txt")
    bible_str = bible_file.read()
    # preprocessing
    # convert to lowercase
    bible_str = bible_str.lower()
    # remove all special characters and numbers except \n
    bible_str = re.sub(r'[^A-Za-z\n]+', ' ', bible_str)
    # write the preprocessed text into the file
    bible_file = open("bible.txt", "w")
    bible_file.write(bible_str)
    # close the file
    bible_file.close()

def train_tokenizer():
    """
    """
    vocab_size=2000
    # Train the SentencePiece tokenizer on our text file
    sp.SentencePieceTrainer.train(input="bible.txt", model_prefix='tokenizer_model', model_type="unigram", vocab_size=vocab_size)

    # load the trained model file in the correct format
    trained_tokenizer_model = tf.io.gfile.GFile('tokenizer_model.model', "rb").read()

    # load the model as a tokenizer that we can use for our models
    tokenizer = tft.SentencepieceTokenizer(
        model=trained_tokenizer_model, out_type=tf.int32, nbest_size=-1, alpha=1, reverse=False,
        add_bos=False, add_eos=False, return_nbest=False, name=None
    )

    return tokenizer

def prepare_data(text, tokenizer,vocab_size, inputlength_m=32): # input_length_m between 32 and 256
    """
    """
    # tokenize the text
    tokens = tokenizer.tokenize(text)

    # create windows
    windowed_tokens = tft.sliding_window(data=tokens, width=inputlength_m+1)
    # the first m window tokens are inputs
    inputs = windowed_tokens[:, :inputlength_m]
    targets = windowed_tokens[:,inputlength_m]
    targets = tf.one_hot(targets, vocab_size)
    # create TensorFlow dataset from the input-target pairs
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    # shuffle, batch and prefetch
    dataset = dataset.shuffle(1000).batch(32).prefetch(4)

    return dataset
