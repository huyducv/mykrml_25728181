# read version from installed package
from importlib.metadata import version
__version__ = version("mykrml_25728181")

from mykrml_25728181 import data, features, models


def main():
    """Entry point for the ``mykrml-25728181`` console script.

    This is a library rather than a command-line tool, so the command just
    reports which version is installed - useful for confirming that the
    published wheel is the one the notebook is importing.
    """
    print(f"mykrml_25728181 {__version__}")


__all__ = ["data", "features", "models", "main", "__version__"]
