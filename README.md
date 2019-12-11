# base_dockex
A collection of Python base classes for [Dockex](https://github.com/ConnexonSystems/dockex) modules. 

Python 3.4+

## Table of Contents

1. [BaseDockex](#BaseDockex)
2. [BaseModel](#BaseModel)
3. [BaseJoblibModel](#BaseJoblibModel)
4. [BaseKerasModel](#BaseKerasModel)

<a name="BaseDockex"></a>
## BaseDockex

Dockex module base class.

This base class reads a JSON file for a Dockex experiment job and extracts
the params, input_pathnames, and output_pathnames.

Subclasses must implement the ```run``` method.

__Arguments__

* **input_args (```list```)**: Input arguments to the module similar to
sys.argv.
Second element must be a string pathname to a JSON file with ```params```,
```input_pathnames```, and ```output_pathnames``` keys.

<a name="BaseModel"></a>
## BaseModel

Dockex model base class. 

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

<a name="BaseJoblibModel"></a>
## BaseJoblibModel

Dockex ```BaseModel``` base class with joblib ```load``` and ```save```
methods.

Subclasses must provide an ```instantiate_model``` method that sets a
```self.model``` with ```self.model.fit``` and ```self.model.predict```
methods.

```BaseJoblibModel``` adds joblib as a requirement.

<a name="BaseKerasModel"></a>
## BaseKerasModel

Dockex ```BaseModel``` base class for [Keras](https://github.com/keras-team/keras)
models.

Subclasses must provide an ```instantiate_model``` method that sets a
```self.model``` with a Keras model.

```BaseKerasModel``` adds tensorflow 2.0 and keras as requirements.
