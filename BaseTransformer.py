import abc
import numpy as np
import random

from .BaseDockex import BaseDockex


class BaseTransformer(BaseDockex):
    """
    Dockex transformer base class.

    This base class defines a transformer interface. Data is
    loaded and saved with numpy. This class exposes ```fit```, ```fit_transform```, ```transform```,
    ```fit_then_transform```, and ```inverse_transform``` methods.

    Subclasses must provide an ```instantiate_transformer``` method that sets a
    ```self.transformer```.

    Subclasses may optionally provide ```load``` and ```save``` methods for
    reading/writing transformers; however, the module's ```transform``` and ```inverse_transform``` method will
    fail without a ```load``` implementation.

    ```BaseTransformer``` adds numpy as a requirement.

    __Parameters__

    * **method (```str```)**: Required. Defines the method of self.transformer to call. ```fit_then_transform```
    will cause ```self.transformer.fit``` to be called followed by ```self.transformer.transform```.

    * **save_transformer (```bool```)**: Set to ```true``` to save the transformer if a
    ```save``` method is implemented.

    * **random_seed (```int```)**: Random seed for numpy.random.seed() and
    random.seed()

    __Input Pathnames__

    * **X_train_npy, X_valid_npy, X_test_npy** (numpy file): Train / valid / test set features.

    * **y_train_npy, y_valid_npy, y_test_npy** (numpy file): Train / valid / test set targets.

    __Output Pathnames__

    * **transform_train_npy, transform_valid_npy, transform_test_npy** (numpy file): Train / valid / test set transformed data.

    """

    def __init__(self, input_args):
        super().__init__(input_args)

        self.method = self.params["method"]
        self.save_transformer = self.params["save_transformer"]
        self.random_seed = self.params["random_seed"]

        # input data
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self.transformer = None

        # output data
        self.transform_train = None
        self.transform_valid = None
        self.transform_test = None

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
    def instantiate_transformer(self):
        """
        This method should set ```self.transformer```.
        """
        pass

    def fit(self):
        print("Fitting transformer")
        if self.y_train is not None:
            self.transformer.fit(self.X_train, y=self.y_train)
        else:
            self.transformer.fit(self.X_train)

    def fit_transform(self):
        print("Fit_transforming transformer")
        if self.y_train is not None:
            self.transform_train = self.transformer.fit_transform(self.X_train, y=self.y_train)
        else:
            self.transform_train = self.transformer.fit_transform(self.X_train)

    def transform(self):
        print("Transforming")
        if self.X_train is not None:
            self.transform_train = self.transformer.transform(self.X_train)

        if self.X_valid is not None:
            self.transform_valid = self.transformer.transform(self.X_valid)

        if self.X_test is not None:
            self.transform_test = self.transformer.transform(self.X_test)

    def inverse_transform(self):
        print("Inverse transforming")
        if self.X_train is not None:
            self.transform_train = self.transformer.inverse_transform(self.X_train)

        if self.X_valid is not None:
            self.transform_valid = self.transformer.inverse_transform(self.X_valid)

        if self.X_test is not None:
            self.transform_test = self.transformer.inverse_transform(self.X_test)

    def load(self):
        raise NotImplementedError

    def save_output_arrays(self):
        print("Writing output arrays")
        if self.transform_train is not None:
            np.save(self.output_pathnames["transform_train_npy"], self.transform_train)

        if self.transform_valid is not None:
            np.save(self.output_pathnames["transform_valid_npy"], self.transform_valid)

        if self.transform_test is not None:
            np.save(self.output_pathnames["transform_test_npy"], self.transform_test)

    def save(self):
        raise NotImplementedError

    def run(self):
        print("Running")

        self.set_random_seeds()

        self.load_input_arrays()

        if self.method == "fit":
            self.instantiate_transformer()
            self.fit()

        elif self.method == "fit_transform":
            self.instantiate_transformer()
            self.fit_transform()

        elif self.method == "fit_then_transform":
            self.instantiate_transformer()
            self.fit()
            self.transform()

        elif self.method == "transform":
            self.load()
            self.transform()

        elif self.method == "inverse_transform":
            self.load()
            self.inverse_transform()

        if self.method != "fit":
            self.save_output_arrays()

        if self.save_transformer is True:
            self.save()

        print("Success")
