import abc
import numpy as np
import random

from .BaseDockex import BaseDockex


class BaseModel(BaseDockex):
    """
    Dockex model base class.

    This base class defines a model interface similar to scikit-learn. Data is
    loaded and saved with numpy. This class exposes ```fit```, ```predict```, ```fit_then_predict```,
    and ```fit_predict``` methods.

    Subclasses must provide an ```instantiate_model``` method that sets a
    ```self.model```.

    Subclasses may optionally provide ```load``` and ```save``` methods for
    reading/writing models; however, the module's ```predict``` method will
    fail without a ```load``` implementation.

    ```BaseModel``` adds numpy as a requirement.

    __Parameters__

    * **method (```str```)**: Required. Defines the method of self.model to call. ```fit_then_predict```
    will cause ```self.model.fit``` to be called followed by ```self.model.predict```.

    * **save_model (```bool```)**: Set to ```true``` to save the model if a
    ```save``` method is implemented.

    * **random_seed (```int```)**: Random seed for numpy.random.seed() and
    random.seed()

    __Input Pathnames__

    * **X_train_npy, X_valid_npy, X_test_npy** (numpy file): Train / valid / test set features.

    * **y_train_npy, y_valid_npy, y_test_npy** (numpy file): Train / valid / test set targets. May be
    1D but will be expanded to 2D.

    __Output Pathnames__

    * **predict_train_npy, predict_valid_npy, predict_test_npy** (numpy file): Train / valid / test set predictions.
    """

    def __init__(self, input_args):
        super().__init__(input_args)

        self.method = self.params["method"]
        self.save_model = self.params["save_model"]
        self.random_seed = self.params["random_seed"]

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
            print("Setting random seeds")
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def safe_load_input_array(self, key, allow_pickle=False):
        if key in self.input_pathnames.keys():
            if self.input_pathnames[key] is not None:
                return np.load(self.input_pathnames[key], allow_pickle=allow_pickle)

        return None

    def load_input_arrays(self):
        print("Loading inputs")
        self.X_train = self.safe_load_input_array('X_train_npy')
        self.y_train = self.safe_load_input_array('y_train_npy', allow_pickle=True)
        self.X_valid = self.safe_load_input_array('X_valid_npy')
        self.y_valid = self.safe_load_input_array('y_valid_npy', allow_pickle=True)
        self.X_test = self.safe_load_input_array('X_test_npy')
        self.y_test = self.safe_load_input_array('y_test_npy', allow_pickle=True)

    @abc.abstractmethod
    def instantiate_model(self):
        """
        This method should set ```self.model```.
        """
        pass

    def fit(self):
        print("Fitting model")
        if self.y_train is not None:
            self.model.fit(self.X_train, self.y_train)
        else:
            self.model.fit(self.X_train)

    def fit_predict(self):
        print("Fit_predicting model")
        if self.y_train is not None:
            self.model.fit_predict(self.X_train, self.y_train)
        else:
            self.model.fit_predict(self.X_train)

    def predict(self):
        print("Predicting")
        if self.X_train is not None:
            self.predict_train = self.model.predict(self.X_train)

        if self.X_valid is not None:
            self.predict_valid = self.model.predict(self.X_valid)

        if self.X_test is not None:
            self.predict_test = self.model.predict(self.X_test)

    def predict_proba(self):
        print("Predicting probabilities")
        if self.X_train is not None:
            self.predict_train = self.model.predict_proba(self.X_train)

        if self.X_valid is not None:
            self.predict_valid = self.model.predict_proba(self.X_valid)

        if self.X_test is not None:
            self.predict_test = self.model.predict_proba(self.X_test)

    def load(self):
        raise NotImplementedError

    def save_output_arrays(self):
        print("Writing output arrays")
        if self.predict_train is not None:
            np.save(self.output_pathnames["predict_train_npy"], self.predict_train)

        if self.predict_valid is not None:
            np.save(self.output_pathnames["predict_valid_npy"], self.predict_valid)

        if self.predict_test is not None:
            np.save(self.output_pathnames["predict_test_npy"], self.predict_test)

    def save(self):
        raise NotImplementedError

    def run(self):
        print("Running")

        self.set_random_seeds()

        self.load_input_arrays()

        if self.method == "fit":
            self.instantiate_model()
            self.fit()

        elif self.method == "fit_predict":
            self.instantiate_model()
            self.fit_predict()

        elif self.method == "fit_then_predict":
            self.instantiate_model()
            self.fit()
            self.predict()

        elif self.method == "fit_then_predict_proba":
            self.instantiate_model()
            self.fit()
            self.predict_proba()

        elif self.method == "predict":
            self.load()
            self.predict()

        elif self.method == "predict_proba":
            self.load()
            self.predict_proba()

        if self.method != "fit":
            self.save_output_arrays()

        if self.save_model is True:
            self.save()

        print("Success")
