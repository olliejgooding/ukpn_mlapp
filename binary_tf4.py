import logging
import re
import string
import time
from typing import Tuple, Union, List, Dict

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

level = logging.INFO
logging.basicConfig(level=level)
logger = logging.getLogger(__name__)

import pandas as pd
import pathlib

path = pathlib.Path(r"C:\Users\ollie\OneDrive\Documents\Coding\UKPN\Scraperdev\traindatav3.xlsx")
df = pd.read_excel(path)
df = df.sample(frac=1)


def get_in_format(df):
    # Load the training data
    train = df.head(int(len(df) * (80 / 100)))
    test = df.tail(int(len(df) * (20 / 100)))
    train_texts = list(train['len'].astype('str'))
    train_labels = list(train['PP_Yes'].astype('int'))
    # Load the validation data.
    test_texts = list(test['len'].astype('str'))
    test_labels = list(test['PP_Yes'].astype('int'))
    return np.array(train_texts), np.array(train_labels), np.array(test_texts), np.array(test_labels)


class TFModel(tf.Module):
    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, ), dtype=tf.string)])
    def prediction(self, review: str) -> Dict[str, Union[str, List[float]]]:
        return {'prediction': self.model(review),
                'description': 'prediction ranges from 0 (negative) to 1 (positive)'}


class ModelTrainer:
    def __init__(self) -> None:
        self.tf_model_wrapper: TFModel

        # Model Architecture parameters
        self.embed_size = 128
        self.max_features = 20000
        self.epochs = 10
        self.batch_size = 128
        self.max_len = 500

    def fetch_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Load your data as needed
        train_texts, train_labels, test_texts, test_labels = get_in_format(df)
        return train_texts, train_labels, test_texts, test_labels


    def custom_preprocessing(self, raw_text: str) -> tf.string:
        lowercase = tf.strings.lower(raw_text)
        stripped_html = tf.strings.regex_replace(lowercase, "\n", ' ')
        return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


    def init_vectorize_layer(self, text_dataset: np.ndarray) -> TextVectorization:
        text_vectorizer = TextVectorization(
            max_tokens=self.max_features,
            standardize=self.custom_preprocessing,
            output_mode='int',
            output_sequence_length=self.max_len
        )
        text_vectorizer.adapt(text_dataset)

        # Explicitly specify the input shape during initialization
        #text_vectorizer.build((1,))  # Assuming 1 is the batch size for inference

        return text_vectorizer


    def init_model(self, text_dataset: np.ndarray) -> tf.keras.Model:
        # Your model initialization logic
        vectorizer = self.init_vectorize_layer(text_dataset)
        raw_input = tf.keras.Input(shape=(1,), dtype=tf.string)
        x = vectorizer(raw_input)
        x = tf.keras.layers.Embedding(self.max_features + 1, self.embed_size)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        op_units, op_activation = 1, 'sigmoid'
        predictions = tf.keras.layers.Dense(op_units, activation=op_activation)(x)
        model = tf.keras.Model(raw_input, predictions)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def train(self) -> None:
        train_texts, train_labels, _, _ = self.fetch_data()
        model = self.init_model(train_texts)
        model.fit(train_texts, train_labels, epochs=self.epochs, batch_size=self.batch_size)
        self.tf_model_wrapper = TFModel(model)
        tf.saved_model.save(self.tf_model_wrapper.model, f'classifier/saved_models/{int(time.time())}',
                            signatures={'serving_default': self.tf_model_wrapper.prediction})
        logger.info('saving SavedModel to classifier/saved_models')

    def train1(self) -> None:
        train_texts, train_labels, _, _ = self.fetch_data()
        model_wrapper = self.init_model(train_texts)
        train_texts_np = np.array(train_texts)
        model_wrapper.model.fit(train_texts_np, train_labels, epochs=self.epochs, batch_size=self.batch_size)

        # Save the model
        timestamp = int(time.time())
        save_path = f'classifier/saved_models/{timestamp}'
        tf.saved_model.save(model_wrapper.model, save_path, signatures={'serving_default': model_wrapper.prediction})
        logger.info(f'SavedModel saved to {save_path}')

        # Save the vectorizer
        vectorizer_save_path = f'classifier/saved_models/{timestamp}/vectorizer'
        model_wrapper.text_vectorizer.save(vectorizer_save_path)
        logger.info(f'TextVectorization saved to {vectorizer_save_path}')


if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.train()
