"""
Main xcell package
==================================

description needed?
"""

import os
from pathlib import Path

from . import util
from . import misc
from . import fem
from . import geometry
from . import elements
from . import meshes
from . import visualizers
from . import io
from . import colors
from . import signals
from .xCell import *


_folderstem = os.path.join(Path.home(),'Documents')

if os.path.exists(_folderstem):
    _folder = os.path.join(_folderstem,'xcell')
else:
    _folder = os.path.join(Path.home(),'xcell')

#: Root directory where all generated data is stored
DATA_DIR = _folder

#TODO: better handling of cross-platform NEURON dependencies
if os.name != "nt":
    from . import nrnutil
