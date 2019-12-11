import abc
import joblib
import warnings

from .BaseModel import BaseModel


class BaseJoblibModel(BaseModel):
    """
    Dockex ```BaseModel``` base class with joblib ```load``` and ```save```
    methods.

    Subclasses must provide an ```instantiate_model``` method that sets a
    ```self.model``` with ```self.model.fit``` and ```self.model.predict```
    methods.

    ```BaseJoblibModel``` adds joblib as a requirement.
    """

    def __init__(self, input_args):
        super().__init__(input_args)

    @abc.abstractmethod
    def instantiate_model(self):
        pass

    def load(self):
        if self.input_pathnames["model_joblib"] is not None:
            print("Loading model")
            self.model = joblib.load(self.input_pathnames["model_joblib"])

        else:
            raise ValueError(
                'Input pathname "model_joblib" must point to a saved model.'
            )

    def save(self):
        if self.method == "predict":
            warnings.warn(
                "User requested save model when model was already loaded from file. Skipping model save."
            )
        else:
            print("Saving model")
            with open(self.output_pathnames["model_joblib"], "wb") as model_file:
                joblib.dump(self.model, model_file)
