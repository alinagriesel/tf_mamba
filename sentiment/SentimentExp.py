import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizer
from einops import rearrange, repeat
from model import *
import matplotlib.pyplot as plt

def data_preprocessing(data, targets, tokenizer, max_length):
  tokenized_data = []
  for input in data:
      t = tokenizer.encode_plus(text=input, max_length=max_length, padding="max_length")
      tokenized_data.append(t["input_ids"])
  binary_targets = tf.math.round(targets)
  dataset = tf.data.Dataset.from_tensor_slices((tokenized_data, binary_targets))
  dataset = dataset.shuffle(1000).batch(32).prefetch(4)
  return dataset

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
    dataset = load_dataset("sst", "default")
   
    # longest sentence 267 -> max seqzence lengtha and input ? -> padding
    train_data = dataset["train"]["sentence"]
    # labels are floats between 0 and 1, need to be rounded to zero or 1 for sentiment classification
    train_labels = dataset["train"]["label"]
    print(len(train_data))

    val_data = dataset["validation"]["sentence"]
    # labels are floats between 0 and 1, need to be rounded to zero or 1 for sentiment classification
    val_labels = dataset["validation"]["label"]
    print(len(val_data))

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    vocab_size = tokenizer.vocab_size

    train_dataset = data_preprocessing(train_data, train_labels, tokenizer, 267)
    validation_dataset = data_preprocessing(val_data, val_labels, tokenizer, 267)

    model = MambaModel(1, vocab_size, input_dim=267)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()

    # compile the model (here, adding a loss function and an optimizer)
    model.compile(optimizer = optimizer, loss=loss, metrics=["accuracy"])

    history = model.fit(train_dataset,validation_data=validation_dataset, epochs=8)

    visualise_results(history)

if __name__ == '__main__':
    run_training()

