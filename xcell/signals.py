"""
Handlers for dynamic signals
"""
import numpy as np


# TODO: implement __getitem__
class Signal:
    """
    Base class for time-dependent values.

    Attributes
    ----------
    value : float
        Constant value of signal.

    """

    def __init__(self, value):
        self.value = value

    def get_value_at_time(self, t):
        """
        Get value of signal at time t.

        Parameters
        ----------
        t : float
            Time to query signal.

        Returns
        -------
        float
            Value at time.
        """
        return self.value

    def reset(self):
        pass


class PiecewiseSignal(Signal):
    """
    Piecewise signal from time, value pairs.
    """

    def __init__(self, t0=0, y0=0):
        self.times = [t0]
        self.values = [y0]
        self._current_index = 0

    def get_value_at_time(self, t):
        is_before = np.less_equal(self.times, t)
        last_index = np.max(np.argwhere(is_before))
        return self.values[last_index]

    def reset(self):
        self._current_index = 0


class BiphasicSignal(PiecewiseSignal):
    def __init__(self, t0=0, y0=0):
        super().__init__(t0, y0)

    def add_pulse(self, t_start, pulse_duration, pulse_amplitude, interphase=0.0):
        """
        Add a biphasic pulse to the signal.

        Parameters
        ----------
        t_start : float
            Time of pulse initiation
        pulse_duration : float
            Duration of a pulse phase in seconds.
        pulse_amplitude : float
            Amplitude of pulse. Use a negative value to place the negative phase first.
        interphase : float
            Time between phases of the pulse (default 0.) Not yet implemented.
        """
        times = [t_start, t_start + pulse_duration, t_start + 2 * pulse_duration]
        vals = [pulse_amplitude, -pulse_amplitude, 0]

        self.times.extend(times)
        self.values.extend(vals)
