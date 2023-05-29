"""
Handlers for dynamic signals
"""
import numpy as np


class Signal:
    def __init__(self, value):
        self._value = value

    def getValueAtTime(self, t):
        return self._value

    def reset(self):
        pass


class PiecewiseSignal(Signal):
    def __init__(self, t0=0, y0=0):
        self.times = [t0]
        self.values = [y0]

        self._currentIdx = 0

    def getValueAtTime(self, t):
        isBefore = np.less_equal(self.times, t)
        lastIdx = np.max(np.argwhere(isBefore))
        return self.values[lastIdx]

        # if self._currentIdx < (len(self.times)-1):
        #     nextTime = self.times[self._currentIdx+1]
        #     if t >= nextTime:
        #         self._currentIdx += 1

        # return self.values[self._currentIdx]

    def reset(self):
        self._currentIdx = 0


class BiphasicSignal(PiecewiseSignal):
    def __init__(self, t0=0, y0=0):
        super().__init__(t0, y0)

    def addPulse(self, tstart, pulseDur, pulseAmp):
        times = [tstart, tstart+pulseDur, tstart+2*pulseDur]
        vals = [pulseAmp, -pulseAmp, 0]

        self.times.extend(times)
        self.values.extend(vals)
