import abc
import numpy as np
import random

from .BaseDockex import BaseDockex


class BaseModel(BaseDockex):
    """
    dockex model base class.

    This base class defines a model interface similar to scikit-learn. Data is
    loaded and saved with numpy. This class exposes ```fit```, ```predict```,
    and ```fit_predict``` methods.

    Subclasses must provide an ```instantiate_model``` method that sets a
    ```self.model``` with ```self.model.fit``` and ```self.model.predict```
    methods.

    Subclasses may optionally provide ```load``` and ```save``` methods for
    reading/writing models; however, the module's ```predict``` method will
    fail without a ```load``` implementation.

    ```BaseModel``` adds numpy as a requirement.

    __Parameters__

    * **method (```str```)**: Required. ```fit``` to train a model.
    ```fit_predict``` to train a model and generate predictions. ```predict```
    to load a previously saved model and generate predictions.

    * **save_model (```bool```)**: Set to ```true``` to save the model if a
    ```save``` method is implemented.

    * **random_seed (```int```)**: Random seed for numpy.random.seed() and
    random.seed()

    __Input Pathnames__

    * **X_train_npy** (numpy file): 2D+ array of training set features.

    * **y_train_npy** (numpy file): 2D array of training set targets. May be
    1D but will be expanded to 2D.

    * **X_valid_npy** (numpy file): 2D+ array of validation set features.

    * **y_valid_npy** (numpy file): 2D array of validation set targets. May be
    1D but will be expanded to 2D.

    * **X_test_npy** (numpy file): 2D+ array of testing set features.

    * **y_test_npy** (numpy file): 2D array of testing set targets. May be
    1D but will be expanded to 2D.

    __Output Pathnames__

    * **predict_train_npy** (numpy file): 2D array of training set predictions.

    * **predict_valid_npy** (numpy file): 2D array of validation set
    predictions.

    * **predict_test_npy** (numpy file): 2D array of testing set predictions.
    """
    
    def __init__(self, input_args):
        super().__init__(input_args)

        self.method = self.params['method']
        self.save_model = self.params['save_model']
        self.random_seed = self.params['random_seed']
        
        # input data
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self.model = None
        
        # output data
        self.predict_train = None
        self.predict_valid = None
        self.predict_test = None

    def set_random_seeds(self):
        if self.random_seed is not None:
            print('Setting random seeds')
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def load_input_arrays(self):
        print('Loading inputs')
        if 'X_train_npy' in self.input_pathnames.keys():
            self.X_train = np.load(self.input_pathnames['X_train_npy'])
        
        if 'y_train_npy' in self.input_pathnames.keys():
            self.y_train = np.load(self.input_pathnames['y_train_npy'], allow_pickle=True)

        if 'X_valid_npy' in self.input_pathnames.keys():
            self.X_valid = np.load(self.input_pathnames['X_valid_npy'])

        if 'y_valid_npy' in self.input_pathnames.keys():
            self.y_valid = np.load(self.input_pathnames['y_valid_npy'], allow_pickle=True)
            
        if 'X_test_npy' in self.input_pathnames.keys():
            self.X_test = np.load(self.input_pathnames['X_test_npy'])

        if 'y_test_npy' in self.input_pathnames.keys():
            self.y_test = np.load(self.input_pathnames['y_test_npy'], allow_pickle=True)

    def check_data_shape(self):
        print('Checking data shape')
        if len(self.y_train.shape) == 1:
            self.y_train = np.expand_dims(self.y_train, 1)
            self.y_valid = np.expand_dims(self.y_valid, 1)
            self.y_test = np.expand_dims(self.y_test, 1)

    @abc.abstractmethod
    def instantiate_model(self):
        """
        This method should set ```self.model``` which should include
        ```self.model.fit``` and ```self.model.predict``` methods.
        """
        pass

    def fit(self):
        self.instantiate_model()

        print('Fitting model')
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        print('Predicting')
        if self.X_train is not None:
            self.predict_train = self.model.predict(self.X_train)

        if self.X_valid is not None:
            self.predict_valid = self.model.predict(self.X_valid)

        if self.X_test is not None:
            self.predict_test = self.model.predict(self.X_test)

    def load(self):
        raise NotImplementedError

    def save_output_arrays(self):
        print('Writing output arrays')
        if self.predict_train is not None:
            np.save(self.output_pathnames['predict_train_npy'], self.predict_train)

        if self.predict_valid is not None:
            np.save(self.output_pathnames['predict_valid_npy'], self.predict_valid)

        if self.predict_test is not None:
            np.save(self.output_pathnames['predict_test_npy'], self.predict_test)

    def save(self):
        raise NotImplementedError

    def run(self):
        print('Running')

        self.set_random_seeds()

        self.load_input_arrays()
        self.check_data_shape()

        if self.method == 'fit':
            self.fit()

        elif self.method == 'fit_predict':
            self.fit()
            self.predict()

        elif self.method == 'predict':
            self.load()
            self.predict()

        if self.method != 'fit':
            self.save_output_arrays()

        if self.save_model is True:
            self.save()

        print('Success')
