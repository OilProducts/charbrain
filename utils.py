class MovingAverage:
    """
    A utility class that tracks numeric values (e.g., rewards) and reports
    both a Simple Moving Average (SMA) and an Exponential Moving Average (EMA).

    The user can specify a 'mode' to determine which moving average is
    primarily represented in the string output. However, both SMA and EMA are
    always maintained internally and are accessible via properties.

    Parameters
    ----------
    window_size : int, optional
        Defines the rolling window size for the SMA, and also
        determines the smoothing factor for the EMA. Default is 10.
    mode : str, optional
        The moving average type primarily used for display. Possible values:
        "simple" or "exponential". Default is 'simple'.

    Attributes
    ----------
    window_size : int
        The rolling window size for SMA, used also to derive the EMA alpha.
    mode : str
        The chosen mode for string representation ("simple" or "exponential").
    alpha : float or None
        The smoothing factor for EMA if mode is 'exponential'.
        Defaults to 2 / (window_size + 1).
    values : list
        Stores up to the last `window_size` values for SMA calculation.
    _ema : float or None
        Tracks the running exponential moving average internally.
    """

    def __init__(self, window_size=10, mode='simple'):
        self.window_size = window_size
        self.mode = mode.lower()

        # Store up to window_size values for SMA calculation.
        self.values = []

        # For EMA, keep a single running value instead of the entire history.
        self._ema = None

        # If the user wants EMA, define alpha from window_size.
        # Even if mode = 'simple', we keep alpha available so we can track EMA if needed.
        self.alpha = 2 / (window_size + 1)

    def add_value(self, value):
        """
        Add a single numeric value (e.g., a reward) to the tracker.
        Updates both the SMA and EMA internally.

        Parameters
        ----------
        value : float
            The new value to incorporate into the moving averages.
        """
        # Update the queue for simple moving average
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        # Update the running EMA
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self.alpha * value + (1 - self.alpha) * self._ema

    @property
    def sma(self):
        """
        Current Simple Moving Average (SMA) over the last `window_size` values.

        Returns
        -------
        float
            The SMA of stored values. Returns 0 if no values have been added.
        """
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def ema(self):
        """
        Current Exponential Moving Average (EMA).

        Returns
        -------
        float
            The running EMA of all added values. Returns 0 if no values have been added.
        """
        if self._ema is None:
            return 0.0
        return self._ema

    def __str__(self):
        """
        String representation showing either the SMA or EMA based on the chosen `mode`.

        Returns
        -------
        str
            A formatted string reporting the chosen moving average.
        """
        if self.mode == 'simple':
            return f"SMA (window={self.window_size}): {self.sma:.3f}"
        elif self.mode == 'exponential':
            return f"EMA (alpha={self.alpha:.3f}): {self.ema:.3f}"
        else:
            return "Invalid mode specified."


# Example usage:
if __name__ == "__main__":
    # Demonstrate usage with mode='simple'
    tracker_sma = MovingAverage(window_size=5, mode='simple')
    for i in range(1, 11):
        tracker_sma.add_value(i)
        print(tracker_sma, "| SMA =", tracker_sma.sma, "| EMA =", tracker_sma.ema)

    print()

    # Demonstrate usage with mode='exponential'
    tracker_ema = MovingAverage(window_size=5, mode='exponential')
    for i in range(1, 11):
        tracker_ema.add_value(i)
        print(tracker_ema, "| SMA =", tracker_ema.sma, "| EMA =", tracker_ema.ema)
