# %%
import numpy as np
import matplotlib.pyplot as plt

stimV = 1.
stimI = 150e-6
stimPhase = 1e-3  # 2ms total biphasic

stimLambda = 10

simDuration = 2
nMicro = 24

# Provide consistently seeded RNG
rngSeed = 0
for ii, c in enumerate('xcell'):
    rngSeed += ord(c) << (7*ii)

rng = np.random.default_rng(rngSeed)

dts = rng.exponential(1/stimLambda, (nMicro, stimLambda*simDuration))


class PiecewiseSignal:
    def __init__(self, t0=0, y0=0):
        self.times = [t0]
        self.values = [y0]

        self._currentIdx = 0

    def getCurrentValue(self, t):
        if self.times[self._currentIdx] <= t:
            self._currentIdx += 1

        return self.values[self._currentIdx]


class BiphasicSignal(PiecewiseSignal):
    def __init__(self, t0=0, y0=0):
        super().__init__(t0, y0)

    def addPulse(self, tstart, pulseDur, pulseAmp):
        times = [tstart, tstart+pulseDur, tstart+2*pulseDur]
        vals = [pulseAmp, -pulseAmp, 0]

        self.times.extend(times)
        self.values.extend(vals)


# Enforce specified miniumum 50ms between pulses
tooSoon = dts < 5e-3
tooSoon[:, 0] = False

dts[tooSoon] = 5e-2
times = np.cumsum(dts, axis=1)
# %%

channels = []

for tstarts in times:
    chan = BiphasicSignal()
    for t in tstarts:
        if t > simDuration:
            continue
        else:
            chan.addPulse(t, pulseDur=stimPhase, pulseAmp=stimPhase)

    channels.append(chan)


f, axes = plt.subplots(nMicro, ncols=1, sharex=True)
axes[0].set_xlim(0, simDuration)

for chan, ax in zip(channels, axes):
    ax.step(chan.times, chan.values, where='post')
    ax.axis('off')

axes[-1].set_xlabel('Time [s]')

plt.show()
# %%
