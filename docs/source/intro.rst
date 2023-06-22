Purpose
==========

``xcell`` aims to provide a simple toolkit for simulating the extracellular space, in conjunction with an intracellular model such as `NEURON <https://neuron.yale.edu>`_. In particular, it's geared toward dynamically adaptive meshing, making larger simulations more computationally manageable.

At present, the software is at an early alpha stage; some modifications may be needed to get it working on your machine. The `gallery examples <auto_examples/index>`_ give an idea of the functionality baked into the library and how to use it; detailed `documentation of the API <xcell/xcell>`_ is also available.

Installing
============

To minimize the chance of compatibility issues, it's best to use the `Anaconda distribution. <https://docs.anaconda.com/free/anaconda/install/>`_

In the directory where you download xcell, create an Anaconda environment by executing ::

    conda env create -f xcell.yml

For even faster installation, instead add Mamba to the base Anaconda environment with ::

	conda install -c conda-forge mamba
and create the environment with ::

	conda env create -n xcell
	mamba env update -n xcell --file xcell.yml

Follow the `installation instructions for NEURON <https://nrn.readthedocs.io/en/latest/install/install_instructions.html>`_ according to your operating system.

Finally, install xcell (in developer mode) with ::

    pip install -e .

(the -e flag can be omitted if you don't want to modify xcell's own code)

Citation
===========

coming soon
