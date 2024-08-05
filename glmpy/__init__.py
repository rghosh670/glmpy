"""Top-level package for glmpy."""

__author__ = """Rohit Ghosh"""
__email__ = "rohitghosh101@gmail.com"
__version__ = "0.1.0"

from loguru import logger

from icecream import install
from icecream import ic
install()
ic.configureOutput(includeContext=False)


from glmpy.core import (
    PoissonModel,
    NBModel,
    ZIPModel,
    ZINBModel,
)


__all__ = [
    "PoissonModel",
    "NBModel",
    "ZIPModel",
    "ZINBModel",
]
