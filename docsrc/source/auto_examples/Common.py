#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common resources for simulations.
"""

import xcell
import numpy as np
from os import path

from neuron import h
from neuron.units import ms, mV
import neuron as nrn

h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)
nrn.load_mechanisms('estimsurvey/')


def makeSynthStudy(folderName,
                   rElec=1e-6,
                   vSource=False,
                   xmax=1e-4):
    """


    Parameters
    ----------
    folderName : TYPE
        DESCRIPTION.
    rElec : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    vSource : TYPE, optional
        DESCRIPTION. The default is False.
    xmax : TYPE, optional
        DESCRIPTION. The default is 1e-4.

    Returns
    -------
    study : TYPE
        DESCRIPTION.
    setup : TYPE
        DESCRIPTION.

    """
    datadir = '/home/benoit/smb4k/ResearchData/Results'
    folder = path.join(datadir, folderName)
    bbox = np.append(-xmax*np.ones(3), xmax*np.ones(3))

    study = xcell.SimStudy(folder, bbox)

    setup = study.newSimulation()

    if vSource:
        setup.addVoltageSource(1.,
                               coords=np.zeros(3),
                               radius=rElec)
    else:
        setup.addCurrentSource(4*np.pi*rElec,
                               coords=np.zeros(3),
                               radius=rElec)

    return study, setup


def runAdaptation(setup, maxdepth=8, meshdensity=0.2, metrics=None):
    if metrics is None:
        metrics = []
        for src in setup.currentSources:
            metrics.append(xcell.makeExplicitLinearMetric(maxdepth, meshdensity,
                                                          origin=src.coords))

    setup.makeAdaptiveGrid(metrics, maxdepth)
    setup.finalizeMesh()
    setup.setBoundaryNodes()
    setup.iterativeSolve(tol=1e-9)


def setParams(testCat, testVal, overrides=None):
    params = {
        'Mesh type': 'adaptive',
        'Element type': 'Admittance',
        'Vsrc?': False,
        'BoundaryFunction': None}

    if overrides is not None:
        params.update(overrides)

    params[testCat] = testVal

    return params


def pairedDepthSweep(study, depthRange, testCat, testVals, rElec=1e-6, sigma=np.ones(3), overrides=None):
    lastmesh = {'numel': 0, 'nX': 0}
    meshnum = 0
    for maxdepth in depthRange:
        for tstVal in testVals:
            params = setParams(testCat, tstVal, overrides)

            setup = study.newSimulation()
            setup.mesh.elementType = params['Element type']
            setup.meshtype = params['Mesh type']
            setup.meshnum = meshnum

            if params['Vsrc?']:
                setup.addVoltageSource(1, np.zeros(3), rElec)
                srcMag = 1.
            else:
                srcMag = 4*np.pi*sigma[0]*rElec
                setup.addCurrentSource(srcMag, np.zeros(3), rElec)

            if params['Mesh type'] == 'uniform':
                newNx = int(np.ceil(lastmesh['numel']**(1/3)))
                nX = newNx+newNx % 2
                setup.makeUniformGrid(newNx+newNx % 2)
                print('uniform, %d per axis' % nX)
            elif params['Mesh type'] == r'equal $l_0$':
                setup.makeUniformGrid(lastmesh['nX'])
            else:
                metricCoef = np.array(2**(-maxdepth*0.2),
                                      dtype=np.float64, ndmin=1)
                setup.makeAdaptiveGrid(np.zeros((1, 3)),                                        np.array(
                    maxdepth, ndmin=1), xcell.generalMetric, coefs=metricCoef)

            setup.mesh.elementType = params['Element type']
            setup.finalizeMesh()

            def boundaryFun(coord):
                r = np.linalg.norm(coord)
                return rElec/(r*np.pi*4)

            if params['BoundaryFunction'] == 'Analytic':
                setup.setBoundaryNodes(boundaryFun)
            elif params['BoundaryFunction'] == 'Rubik0':
                setup.setBoundaryNodes(None,
                                       expand=True,
                                       sigma=1.)
            else:
                setup.setBoundaryNodes()
            # setup.getEdgeCurrents()

            # v=setup.solve()
            v = setup.iterativeSolve(None, 1e-9)

            setup.applyTransforms()

            setup.getMemUsage(True)

            errEst, err, ana, _, _ = setup.calculateErrors()
            FVU = xcell.misc.FVU(ana, err)

            minel = setup.mesh.getL0Min()

            print('error: %g' % errEst)

            dataCats = ['Error', 'FVU', 'l0min', testCat]
            dataVals = [errEst, FVU, minel, tstVal]

            data = xcell.misc.getErrorEstimates(setup)

            for k, v in data.items():
                dataCats.append(k)
                dataVals.append(v)

            study.newLogEntry(dataCats, dataVals)
            study.saveData(setup)
            if params['Mesh type'] == 'adaptive':
                lastmesh = {'numel': len(setup.mesh.elements),
                            'nX': setup.ptPerAxis-1}


class Cell:
    def __init__(self, gid, x, y, z, theta):
        self._gid = gid
        self._setup_morphology()

        if 'soma' in dir(self):
            self.all = self.soma.wholetree()
        else:
            self.all = h.allsec()
        self._setup_biophysics()
        self.x = self.y = self.z = 0
        h.define_shape()
        self._rotate_z(theta)
        self._set_position(x, y, z)
        self.vrest = -65
        self._ncs = []

    def __repr__(self):
        return '{}[{}]'.format(self.name, self._gid)

    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(i,
                               x - self.x + sec.x3d(i),
                               y - self.y + sec.y3d(i),
                               z - self.z + sec.z3d(i),
                               sec.diam3d(i))
        self.x, self.y, self.z = x, y, z

    def _rotate_z(self, theta):
        """Rotate the cell about the Z axis."""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))

    def rotateLocal(self, theta):
        """
        Rotate cell about its soma

        """
        x0 = self.soma.x3d(0)
        y0 = self.soma.y3d(0)

        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i) - x0
                y = sec.y3d(i) - y0
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s + x0
                yprime = x * s + y * c + y0
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))

    def forceZ0(self):
        """Hack to fix neurons floating away from plane"""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)

                sec.pt3dchange(i, x, y, 0, sec.diam3d(i))


class BallAndStick(Cell, xcell.nrnutil.RecordedCell):
    name = 'BallAndStick'

    def __init__(self, gid, x, y, z, theta):
        super().__init__(gid, x, y, z, theta)
        r = self.soma.diam/2
        dx = r*np.cos(theta)
        dy = r*np.sin(theta)

        self._set_position(self.x-dx, self.y-dy, self.z)

        self.attachMembraneRecordings([self.soma])
        self.attachSpikeDetector(self.soma)

    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003    # Leak conductance in S/cm2
            seg.hh.el = -54.3     # Reversal potential in mV
        # Insert passive current in the dendrite
        self.dend.insert('pas')
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65    # Leak reversal potential mV

        # NEW: the synapse
        self.syn = h.ExpSyn(self.dend(0.5))
        self.syn.tau = 2 * ms


class Axon10(Cell, xcell.nrnutil.RecordedCell):
    name = 'axon10'

    GAP = 2.5e-4  # width of node of Ranvier in cm
    DIAM = 10  # fiber diameter in um including myelin
    SDD = 0.7  # axon diameter is SDD*DIAM
    ELD = 100  # length of internode/fiber diameter

    RHOI = 100  # cytoplasmic resistivity in ohm cm
    CM = 2  # specific membrane capacitance in uf/cm2

    def __init__(self, gid, x, y, z, nnodes, segL=None):
        self.nnodes = nnodes

        if segL is not None:
            self.ELD = segL

        # super().__init__(gid, x, y, z,0)
        self._gid = gid
        self._setup_morphology()
        self._setup_biophysics()

        self.x = self.y = self.z = 0
        h.define_shape()

        origin = xcell.nrnutil.returnSegmentCoordinates(
            self.nodes[nnodes//2], inMicrons=True)

        self._set_position(x - origin[0],
                           y - origin[1],
                           z - origin[2])

        self.attachSpikeDetector(self.nodes[nnodes//2])

        self.vrest = -70

    def _setup_morphology(self):
        self.nodes = [h.Section(name='node%d' % n, cell=self)
                      for n in range(self.nnodes)]
        self.internodes = [h.Section(name='internode%d' % n, cell=self)
                           for n in range(self.nnodes-1)]

        for node, internode, ii in zip(self.nodes, self.internodes, range(self.nnodes)):
            node.diam = self.SDD*self.DIAM
            node.L = self.GAP*1e4

            internode.diam = self.SDD*self.DIAM
            internode.L = self.ELD*self.DIAM

            internode.connect(node)

            if ii > 0:
                node.connect(lastInternode)

            lastInternode = internode

        self.nodes[-1].connect(self.internodes[-1])

        self.all = h.SectionList(h.allsec())

        # for nn in self.nodes:
        #     nn.diam = 7.
        #     nn.L = 2.5

        # for ii in self.internodes:
        #     ii.diam = 7.
        #     ii.L = 1000.
        #     ii.nseg = 3

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = self.RHOI    # Axial resistance in Ohm * cm
        for sec in self.nodes:
            sec.cm = self.CM      # Membrane capacitance in micro Farads / cm^2
            sec.insert('fh')
            for seg in sec:
                seg.pnabar_fh = 8e-3
                seg.pkbar_fh = 1.2e-3
                seg.ppbar_fh = 0.54e-3
                seg.gl_fh = 0.0303
                seg.nai = 13.74
                seg.nao = 114.5
                seg.ki = 120
                seg.ko = 2.

        for sec in self.internodes:
            sec.cm = 1e-6

        h.celsius = 20.


class MRG(Cell, xcell.nrnutil.RecordedCell):
    name = 'MRG'
    RHOA = 0.7e6
    CM = 0.1
    GM = 1e-3

    PARALENGTH1 = 3
    NODELENGTH = 1.0
    SPACE_P1 = 2e-3
    SPACE_P2 = 4e-3
    SPACE_I = 4e-3

    def __init__(self, gid, x, y, z, theta, fiberD=10.,
                 axonNodes=21):
        nrn.load_mechanisms('../../3810/', False)

        self.fiberD = fiberD
        self.axonNodes = axonNodes
        self.paraNodes1 = 2*(axonNodes-1)
        self.paraNodes2 = 2*(axonNodes-1)
        self.axonInter = 6*(axonNodes-1)
        self.axontotal = 11*axonNodes-10
        self._deps(fiberD)

        super().__init__(gid, x, y, z, theta)

        self.vrest = -80

        self.attachSpikeDetector(self.nodes[axonNodes//2])

    def _deps(self, fiberD):
        if fiberD == 5.7:
            self.g = 0.605
            self.axonD = 3.4
            self.nodeD = 1.9
            self.paraD1 = 1.9
            self.paraD2 = 3.4
            self.deltax = 500
            self.paralength2 = 35
            self.nl = 80
        if fiberD == 7.3:
            self.g = 0.630
            self.axonD = 4.6
            self.nodeD = 2.4
            self.paraD1 = 2.4
            self.paraD2 = 4.6
            self.deltax = 750
            self.paralength2 = 38
            self.nl = 100
        if fiberD == 8.7:
            self.g = 0.661
            self.axonD = 5.8
            self.nodeD = 2.8
            self.paraD1 = 2.8
            self.paraD2 = 5.8
            self.deltax = 1000
            self.paralength2 = 40
            self.nl = 110
        if fiberD == 10.0:
            self.g = 0.690
            self.axonD = 6.9
            self.nodeD = 3.3
            self.paraD1 = 3.3
            self.paraD2 = 6.9
            self.deltax = 1150
            self.paralength2 = 46
            self.nl = 120
        if fiberD == 11.5:
            self.g = 0.700
            self.axonD = 8.1
            self.nodeD = 3.7
            self.paraD1 = 3.7
            self.paraD2 = 8.1
            self.deltax = 1250
            self.paralength2 = 50
            self.nl = 130
        if fiberD == 12.8:
            self.g = 0.719
            self.axonD = 9.2
            self.nodeD = 4.2
            self.paraD1 = 4.2
            self.paraD2 = 9.2
            self.deltax = 1350
            self.paralength2 = 54
            self.nl = 135
        if fiberD == 14.0:
            self.g = 0.739
            self.axonD = 10.4
            self.nodeD = 4.7
            self.paraD1 = 4.7
            self.paraD2 = 10.4
            self.deltax = 1400
            self.paralength2 = 56
            self.nl = 140
        if fiberD == 15.0:
            self.g = 0.767
            self.axonD = 11.5
            self.nodeD = 5.0
            self.paraD1 = 5.0
            self.paraD2 = 11.5
            self.deltax = 1450
            self.paralength2 = 58
            self.nl = 145
        if fiberD == 16.0:
            self.g = 0.791
            self.axonD = 12.7
            self.nodeD = 5.5
            self.paraD1 = 5.5
            self.paraD2 = 12.7
            self.deltax = 1500
            self.paralength2 = 60
            self.nl = 150

        self.Rpn0 = (self.RHOA*.01)/(np.pi *
                                     ((((self.nodeD/2)+self.SPACE_P1) ** 2)-((self.nodeD/2) ** 2)))
        self.Rpn1 = (self.RHOA*.01)/(np.pi *
                                     ((((self.paraD1/2)+self.SPACE_P1) ** 2)-((self.paraD1/2) ** 2)))
        self.Rpn2 = (self.RHOA*.01)/(np.pi *
                                     ((((self.paraD2/2)+self.SPACE_P2) ** 2)-((self.paraD2/2) ** 2)))
        self.Rpx = (self.RHOA*.01)/(np.pi *
                                    ((((self.axonD/2)+self.SPACE_I) ** 2)-((self.axonD/2) ** 2)))
        self.interlength = (self.deltax-self.NODELENGTH -
                            (2*self.PARALENGTH1)-(2*self.paralength2))/6

    def _setup_morphology(self):
        self.nodes = [h.Section(name='node%d' % n, cell=self)
                      for n in range(self.axonNodes)]

        self.MYSA = [h.Section(name='MYSA%d' % n, cell=self)
                     for n in range(self.paraNodes1)]

        self.FLUT = [h.Section(name='FLUT%d' % n, cell=self)
                     for n in range(self.paraNodes2)]
        self.STIN = [h.Section(name='STIN%d' % n, cell=self)
                     for n in range(self.axonInter)]

        for ii in range(self.axonNodes-1):
            chain = [self.nodes[ii],
                     self.MYSA[2*ii],
                     self.FLUT[2*ii]]
            chain.extend([self.STIN[6*ii+n] for n in range(6)])
            chain.extend([self.FLUT[2*ii+1],
                          self.MYSA[2*ii+1],
                          self.nodes[ii+1]])

            for jj in range(len(chain)-1):
                chain[jj+1].connect(chain[jj])

    def _setup_biophysics(self):
        for sec in self.nodes:
            sec.insert('axnode')
            sec.insert('extracellular')
            sec.nseg = 1
            sec.diam = self.nodeD
            sec.L = self.NODELENGTH
            sec.Ra = self.RHOA/1e4
            sec.cm = 2
            sec.xraxial[0] = self.Rpn0
            sec.xg[0] = 1e10
            sec.xc[0] = 0

        for sec in self.MYSA:
            sec.insert('pas')
            sec.insert('extracellular')
            ratio = self.paraD1/self.fiberD

            sec.nseg = 1
            sec.diam = self.fiberD
            sec.L = self.PARALENGTH1
            sec.Ra = self.RHOA*(1/(ratio)**2)/1e4
            sec.cm = 2*ratio

            sec.g_pas = 1e-3 * ratio
            sec.e_pas = -80

            sec.xraxial[0] = self.Rpn1
            sec.xg[0] = self.GM/(2*self.nl)
            sec.xc[0] = self.CM/(2*self.nl)

        for sec in self.FLUT:
            sec.insert('pas')
            sec.insert('extracellular')
            ratio = self.paraD2/self.fiberD

            sec.nseg = 1
            sec.diam = self.fiberD
            sec.L = self.paralength2
            sec.Ra = self.RHOA*(1/(ratio)**2)/1e4
            sec.cm = 2*ratio

            sec.g_pas = 1e-3 * ratio
            sec.e_pas = -80

            sec.xraxial[0] = self.Rpn2
            sec.xg[0] = self.GM/(2*self.nl)
            sec.xc[0] = self.CM/(2*self.nl)

        for sec in self.STIN:
            sec.insert('pas')
            sec.insert('extracellular')
            ratio = self.axonD/self.fiberD

            sec.nseg = 1
            sec.diam = self.fiberD
            sec.L = self.interlength
            sec.Ra = self.RHOA*(1/(ratio)**2)/1e4
            sec.cm = 2*ratio

            sec.g_pas = 1e-3 * ratio
            sec.e_pas = -80

            sec.xraxial[0] = self.Rpx
            sec.xg[0] = self.GM/(2*self.nl)
            sec.xc[0] = self.CM/(2*self.nl)


class Ring:
    """A network of *N* ball-and-stick cells where cell n makes an
    excitatory synapse onto cell n + 1 and the last, Nth cell in the
    network projects to the first cell.
    """

    def __init__(self, N=5, stim_w=0.02, stim_t=4, stim_delay=1, syn_w=0.01, syn_delay=5, r=50, dendSegs=1):
        """
        :param N: Number of cells.
        :param stim_w: Weight of the stimulus
        :param stim_t: time of the stimulus (in ms)
        :param stim_delay: delay of the stimulus (in ms)
        :param syn_w: Synaptic weight
        :param syn_delay: Delay of the synapse
        :param r: radius of the network
        """
        self._syn_w = syn_w
        self._syn_delay = syn_delay
        self._create_cells(N, r)
        self._connect_cells()
        # add stimulus
        self._netstim = h.NetStim()
        self._netstim.number = 1
        self._netstim.start = stim_t
        self._nc = h.NetCon(self._netstim, self.cells[0].syn)
        self._nc.delay = stim_delay
        self._nc.weight[0] = stim_w

        for cell in self.cells:
            cell.dend.nseg = dendSegs
            h.define_shape()
            cell.forceZ0()

    def _create_cells(self, N, r):
        self.cells = []
        for i in range(N):
            theta = i * 2 * h.PI / N
            newCell = BallAndStick(
                i, h.cos(theta) * r, h.sin(theta) * r, 0, theta)
            newCell.rotateLocal(-h.PI/2)
            self.cells.append(newCell)

    def _connect_cells(self):
        for source, target in zip(self.cells, self.cells[1:] + [self.cells[0]]):
            nc = h.NetCon(source.soma(0.5)._ref_v, target.syn, sec=source.soma)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            source._ncs.append(nc)
