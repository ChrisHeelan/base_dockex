import abc
import json


class BaseDockex(abc.ABC):
    """
    dockex module base class.

    This base class reads a JSON file for a dockex experiment job and extracts
    the params, input_pathnames, and output_pathnames.

    Subclasses must implement the ```run``` method.

    __Arguments__

    * **input_args (```list```)**: Input arguments to the module similar to
    sys.argv.
    Second element must be a string pathname to a JSON file with ```params```,
    ```input_pathnames```, and ```output_pathnames``` keys.
    """

    def __init__(self, input_args):
        super().__init__()

        with open(input_args[1], "r") as f:
            self._config = json.load(f)

        self._params = self._config["params"]
        self._input_pathnames = self._config["input_pathnames"]
        self._output_pathnames = self._config["output_pathnames"]

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        raise AttributeError('Must be set through JSON file')

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        raise AttributeError('Must be set through JSON file')

    @property
    def input_pathnames(self):
        return self._input_pathnames

    @input_pathnames.setter
    def input_pathnames(self, value):
        raise AttributeError('Must be set through JSON file')

    @property
    def output_pathnames(self):
        return self._output_pathnames

    @output_pathnames.setter
    def output_pathnames(self, value):
        raise AttributeError('Must be set through JSON file')

    @abc.abstractmethod
    def run(self):
        """
        Subclasses should be executed using this method.
        """
        raise NotImplementedError
