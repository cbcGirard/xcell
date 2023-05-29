# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from xcell.signals import BiphasicSignal
stimV = 1.
stimI = 150e-6
stimPhase = 1e-3  # 2ms total biphasic

stimLambda = 10

simDuration = 2
minDt = 50e-3

# nMicro = 9

# Provide consistently seeded RNG
rngSeed = 0
for ii, c in enumerate('xcell'):
    rngSeed += ord(c) << (7*ii)

rng = np.random.default_rng(rngSeed)


def getSignals(nMicro, step=None):
    dts = rng.exponential(1/stimLambda, (nMicro, stimLambda*simDuration))

    # Enforce specified miniumum 50ms between pulses
    tooSoon = dts < minDt
    tooSoon[:, 0] = False

    dts[tooSoon] = 5e-2
    times = np.cumsum(dts, axis=1)

    # Rounding to simplify simulation
    # step = minDt
    if step is not None:
        times = np.round(times/step)*step

    channels = []

    for tstarts in times:
        chan = BiphasicSignal()
        for t in tstarts:
            if t >= (simDuration - 2e-2):
                continue
            else:
                chan.addPulse(t, pulseDur=stimPhase, pulseAmp=stimI)

        channels.append(chan)

    return channels


if __name__ == '__main__':
    nMicro = 24
    channels = getSignals(nMicro)
    f, axes = plt.subplots(nMicro, ncols=1, sharex=True)
    axes[0].set_xlim(0, simDuration)

    times = []
    for chan, ax in zip(channels, axes):
        ax.step(chan.times, chan.values, where='post')
        ax.axis('off')
        times.extend(chan.times)

    axes[-1].set_xlabel('Time [s]')

    plt.show()

    pickle.dump(channels, open('pattern.xstim', 'wb'))

    print('%d steps' % np.unique(times).shape)
