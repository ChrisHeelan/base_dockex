import abc
import joblib
import warnings

from .BaseTransformer import BaseTransformer


class BaseJoblibTransformer(BaseTransformer):
    """
    Dockex ```BaseTransformer``` base class with joblib ```load``` and ```save```
    methods.

    Subclasses must provide an ```instantiate_transformer``` method that sets a
    ```self.transformer```.

    ```BaseJoblibTransformer``` adds joblib as a requirement.
    """

    def __init__(self, input_args):
        super().__init__(input_args)

    @abc.abstractmethod
    def instantiate_transformer(self):
        pass

    def load(self):
        if self.input_pathnames["transformer_joblib"] is not None:
            print("Loading transformer")
            self.transformer = joblib.load(self.input_pathnames["transformer_joblib"])

        else:
            raise ValueError(
                'Input pathname "transformer_joblib" must point to a saved transformer.'
            )

    def save(self):
        if self.method in ["transform", "inverse_transform"]:
            warnings.warn(
                "User requested save transformer when transformer was already loaded from file. Skipping transformer save."
            )
        else:
            print("Saving transformer")
            with open(self.output_pathnames["transformer_joblib"], "wb") as transformer_file:
                joblib.dump(self.transformer, transformer_file)
