"""
Main xcell package
==================================

description needed?
"""

import os

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

if os.name != "nt":
    from . import nrnutil
