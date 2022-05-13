"""
Taken from the Python scripting tutorial in NEURON's documentation,
https://neuronsimulator.github.io/nrn/tutorials/ball-and-stick-4.html
"""

from neuron import h
h.nrnmpi_init()       # initialize MPI
pc = h.ParallelContext()
print('I am {} of {}'.format(pc.id(), pc.nhost()))
h.quit()              # necessary to avoid a warning message on parallel exit on some systems
