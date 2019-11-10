# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate, BatchNormalization
from keras.models import Model, load_model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class CNNSequenceClassifier:
    """
    A CNN based classifier for text or other sequences.
    """
    def __init__(self, X, Y, epochs: int, embedding_dimension: int = 11 , batch_size: int = 64):
        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(X)
        word_index = tokenizer.word_index
        X = tokenizer.texts_to_sequences(X)
        # Set max length based on previously visualized sequence distributions
        self.max_sequence_length = sum(map(len, X)) // len(X)
        X = sequence.pad_sequences(X, maxlen=self.max_sequence_length)

        # Transform class labels to one-hot encodings
        lb = LabelBinarizer()
        Y = lb.fit_transform(Y)

        filter_sizes = (3,5,9,15,21)
        conv_blocks = []

        embedding_layer = Embedding(
            len(tokenizer.word_index)+1,
            embedding_dimension,
            input_length=self.max_sequence_length
        )

        callbacks = [EarlyStopping(monitor='val_accuracy', verbose=1, patience=4)]

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        reshape = Dropout(0.1)(embedded_sequences)

        # Add convolutional layer for each filter size
        for size_val in filter_sizes:
            conv = Conv1D(
                filters=32,
                kernel_size=size_val,
                padding='valid',
                activation='relu',
                strides=1)(reshape)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        merged = Concatenate()(conv_blocks)
        dropout = Dropout(0.25)(merged)
        normalize = BatchNormalization()(dropout)
        output = Dense(256, activation='relu')(normalize)
        normalized_output = BatchNormalization()(output)
        predictions = Dense(4, activation='softmax')(normalized_output)
        self.model = Model(sequence_input, predictions)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.model.fit(
            X, Y,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.15,
            callbacks=callbacks,
            epochs=epochs
        )

    def predict(self, X):
        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(X)
        word_index = tokenizer.word_index
        X = tokenizer.texts_to_sequences(X)
        # Set max length based on previously visualized sequence distributions
        X = sequence.pad_sequences(X, maxlen=self.max_sequence_length)

        return self.model.predict(X)

    def load(self, filename: str):
        self.model = load_model(filename)

    def save(self, filename: str):
        self.model.save_weights(filename)
