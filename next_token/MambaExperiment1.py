# imports
import tensorflow as tf
import numpy as np

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from einops import rearrange, repeat
import tensorflow_text as tft

#from google.colab import files
import json

import matplotlib.pyplot as plt
import sentencepiece as sp
import re
import keras 

from model import MambaModel, MambaResBlock, SelectiveSSM

def data_cleaning():
    """
    clean the bible data file by removing special characters and converting all letters to lower case.
    cleaned file is saved in the same directory.
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
    Sentencepiece tokenizer is trained on the bible dataset with vocabulary size of 2000.
    returns:
        trained tokenizer
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
    Parallel preprocessing for some dataset. Data is windowed according to given input length and last token of 
    windowed sequence is set as the target. Targets are one hot encoded and data is batched, prefetched.
    args:
        text: data as string
        tokenizer: trained tokenize to tokenize input data
        vocab_size: needed for one hot encoding
        inputlength_m: sequence length for windowing

    returns:
        train_dataset: 70% of input data 
        val_dataset: 30% of input data
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
    train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.7, right_size=0.3, shuffle=True)
    # shuffle, batch and prefetch
    train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(4)
    val_dataset = val_dataset.shuffle(1000).batch(32).prefetch(4)


    return train_dataset, val_dataset

def visualise_results(history):

  plt.plot(history.history["loss"], label="train loss")
  plt.plot(history.history["val_loss"], label="validation loss")
  plt.title("Train and Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()

  plt.plot(history.history["accuracy"], label="train accuracy")
  plt.plot(history.history["val_accuracy"], label="validation accuracy")
  plt.title("Train and Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()


# Main
def run_training():

    data_cleaning()
    tokenizer = train_tokenizer()

    vocab_size = 2000
    sequence_length = 128

    # shorten dataset
    bible_file = open("bible.txt")
    bible = bible_file.read()
    bible = bible[:len(bible)//14]


    train_data, val_data = prepare_data(bible, tokenizer,vocab_size,inputlength_m = sequence_length)

    model = MambaModel(1, vocab_size, 128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy()

    # compile the model (here, adding a loss function and an optimizer)
    model.compile(optimizer = optimizer, loss=loss, metrics=["accuracy"])

    history = model.fit(train_data,validation_data=val_data, epochs=10)

    visualise_results(history)

if __name__ == '__main__':
    run_training()
