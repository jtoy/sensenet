import distutils.version
import logging
import os
import sys

from sensenet import error
from sensenet.configuration import logger_setup, undo_logger_setup
from sensenet.utils import reraise
from sensenet.version import VERSION as __version__

logger = logging.getLogger(__name__)

# Do this before importing any other gym modules, as most of them import some
# dependencies themselves.
def sanity_check_dependencies():
    import numpy
    import pybullet
    #import requests
    #import six

    if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion('1.10.4'):
        logger.warn("You have 'numpy' version %s installed, but 'sensenet' requires at least 1.10.4. HINT: upgrade via 'pip install -U numpy'.", numpy.__version__)

    #if distutils.version.LooseVersion(requests.__version__) < distutils.version.LooseVersion('2.0'):
    #    logger.warn("You have 'requests' version %s installed, but 'sensenet' requires at least 2.0. HINT: upgrade via 'pip install -U requests'.", requests.__version__)

# We automatically configure a logger with a simple stderr handler. If
# you'd rather customize logging yourself, run undo_logger_setup.
#
# (Note: this code runs before importing the rest of gym, since we may
# print a warning at load time.)
#
# It's generally not best practice to configure the logger in a
# library. We choose to do so because, empirically, many of our users
# are unfamiliar with Python's logging configuration, and never find
# their way to enabling our logging. Users who are aware of how to
# configure Python's logging do have to accept a bit of incovenience
# (generally by caling `gym.undo_logger_setup()`), but in exchange,
# the library becomes much more usable for the uninitiated.
#
# Gym's design goal generally is to be simple and intuitive, and while
# the tradeoff is definitely not obvious in this case, we've come down
# on the side of auto-configuring the logger.

if not os.environ.get('SENSENET_NO_LOGGER_SETUP'):
    logger_setup()
del logger_setup

sanity_check_dependencies()

from sensenet.core import SenseEnv, Space, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
#from sensenet.benchmarks import benchmark_spec
from sensenet.envs import make, spec
#from sensenet.scoreboard.api import upload
#from sensenet import wrappers

__all__ = ["SenseEnv", "Space", "Wrapper", "make", "spec", "upload", "wrappers"]
__all__ = ["SenseEnv", "Space", "Wrapper", "make", "spec", "wrappers"]
