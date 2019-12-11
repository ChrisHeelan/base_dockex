import abc
import warnings
import numpy as np
import random
import tensorflow as tf
from keras.models import load_model

from .BaseModel import BaseModel


class BaseKerasModel(BaseModel):
    """
    Dockex ```BaseModel``` base class for [Keras](https://github.com/keras-team/keras)
    models.

    Subclasses must provide an ```instantiate_model``` method that sets a
    ```self.model``` with a Keras model.

    ```BaseKerasModel``` adds tensorflow 2.0 and keras as requirements.
    """

    def __init__(self, input_args):
        super().__init__(input_args)

        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']

        self.callbacks = []

    def set_random_seeds(self):
        if self.random_seed is not None:
            print('Setting random seeds')
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)

    @abc.abstractmethod
    def instantiate_model(self):
        pass

    def fit(self):
        self.instantiate_model()

        if self.X_valid is not None and self.y_valid is not None:
            validation_data = (self.X_valid, self.y_valid)
        else:
            validation_data = None

        print('Fitting model')
        self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=self.callbacks,
            verbose=2
        )

    def load(self):
        if self.input_pathnames['model_keras'] is not None:
            print('Loading model')
            self.model = load_model(self.input_pathnames['model_keras'])

        else:
            raise ValueError('Input pathname "model_keras" must point to a saved model.')

    def save(self):
        if self.method == 'predict':
            warnings.warn("User requested save model when model was already loaded from file. Skipping model save.")
        else:
            print('Saving model')
            self.model.save(self.output_pathnames['model_keras'])
