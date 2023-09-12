Purpose
==========

``xcell`` aims to provide a simple toolkit for simulating the extracellular space, in conjunction with an intracellular model such as `NEURON <https://neuron.yale.edu>`_. In particular, it's geared toward dynamically adaptive meshing, making larger simulations more computationally manageable.

At present, the software is at an early alpha stage; some modifications may be needed to get it working on your machine. The `gallery examples <auto_examples/index>`_ give an idea of the functionality baked into the library and how to use it; detailed `documentation of the API <autoapi/index.html>`_ is also available.

Installing
============

To minimize the chance of compatibility issues (and take advantage of much faster installers), it's best to use the `Mambaforge  distribution <https://github.com/conda-forge/miniforge#mambaforge>`_ instead of a standard Anaconda distribution, though that should be possible as well (using ```conda``` in place of ```mamba```. MacOS and Windows installations are not yet fully tested, but no further steps beyond this guide should be needed for Linux.

In the directory where you download xcell, create an environment by executing ::

	mamba env create -f xcell

Follow the `installation instructions for NEURON <https://nrn.readthedocs.io/en/latest/install/install_instructions.html>`_ according to your operating system.

Finally, install xcell (in developer mode) with ::

    pip install -e .

(the -e flag can be omitted if you don't want to modify xcell's own code)

Citation
===========

coming soon
