from neo.io import Spike2IO
import os
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi, iirnotch, tf2sos


def highpass(data, fs: float, cutoff_freq: float = 1.0):
    """
    Apply a causal high-pass filter to each channel of a multi-signal numpy array.

    :param data: Multi-signal numpy array where each row represents a channel.
    :param sampling_freq: Sampling frequency of the signal.
    :param cutoff_freq: Cutoff frequency of the high-pass filter.
    :return: Filtered data.
    """
    # Design a second-order sections high-pass Butterworth filter
    sos = butter(N=3, Wn=cutoff_freq, btype="highpass", fs=fs, output="sos")

    # Initialize filtered data array
    filtered_data = np.zeros_like(data)

    # Apply the filter to each channel
    for i in range(data.shape[0]):
        # Set initial state of the filter based on the first value of the channel
        zi = sosfilt_zi(sos) * data[i, 0]
        # Apply the filter
        filtered_data[i, :] = sosfiltfilt(sos, data[i, :])

    return filtered_data


def bandpass(data, fs: float, band_freq: tuple[float, float] = (49, 51)):
    """
    Apply a causal band-pass filter to each channel of a multi-signal numpy array.
    """

    # Design a band-pass Butterworth filter
    sos = butter(N=3, Wn=band_freq, btype="bandpass", fs=fs, output="sos")

    # Initialize filtered data array
    filtered_data = np.zeros_like(data)

    # Apply the filter to each channel
    for i in range(data.shape[0]):
        # Apply the filter
        filtered_data[i, :] = sosfiltfilt(sos, data[i, :])

    return filtered_data


def causal_bandstop(
    data,
    fs: float,
    stopband_freq: tuple[float, float] = (49, 51),
):
    """
    Apply a causal band-stop filter to each channel of a multi-signal numpy array.

    :param data: Multi-signal numpy array where each row represents a channel.
    :param fs: Sampling frequency of the signal.
    :param stopband_freq: Frequency band to be notched out.
    :param quality_factor: Quality factor for the notch filter, which determines the bandwidth around the notch frequency.
    :return: Filtered data.
    """
    # Design a notch filter and convert to SOS format
    sos = butter(N=4, Wn=stopband_freq, fs=fs, btype="bandstop", output="sos")

    # Initialize filtered data array
    filtered_data = np.zeros_like(data)

    # Apply the filter to each channel
    for i in range(data.shape[0]):
        # Set initial state of the filter based on the first value of the channel
        zi = sosfilt_zi(sos) * data[i, 0]
        # Apply the filter
        filtered_data[i, :], _ = sosfilt(sos, data[i, :], zi=zi)

    return filtered_data


def causal_notch(data, fs: float, notch_freq: float = 50, quality_factor: float = 30):
    """
    Apply a causal notch filter to each channel of a multi-signal numpy array.

    :param data: Multi-signal numpy array where each row represents a channel.
    :param fs: Sampling frequency of the signal.
    :param notch_freq: Frequency to be notched out.
    :param quality_factor: Quality factor for the notch filter, which determines the bandwidth around the notch frequency.
    :return: Filtered data.
    """
    # Design a notch filter and convert to SOS format
    b, a = iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    sos = tf2sos(b, a)

    # Initialize filtered data array
    filtered_data = np.zeros_like(data)

    # Apply the filter to each channel
    for i in range(data.shape[0]):
        # Set initial state of the filter based on the first value of the channel
        zi = sosfilt_zi(sos) * data[i, 0]
        # Apply the filter
        filtered_data[i, :], _ = sosfilt(sos, data[i, :], zi=zi)

    return filtered_data


class SmrImporter:
    """Reads signals and metadata (sampling frequency ``fs``, channel names ``ch_names``) from SMR file"""

    def __init__(self, fname: os.PathLike | str):
        analog_signal = (
            Spike2IO(filename=str(fname)).read()[0].segments[0].analogsignals[0]
        )

        self.events = Spike2IO(filename=str(fname)).read()[0].segments[0].events

        self.event_names = [event.name for event in self.events]
        print(self.event_names)

        self.ch_names = analog_signal.array_annotations["channel_names"].tolist()

        print(self.ch_names)

        self.ch_dict = {name: i for i, name in enumerate(self.ch_names)}

        self.fs = analog_signal.sampling_rate.magnitude

        self.data = analog_signal.magnitude.T

    def get_event_timestamps(self, event_name: str) -> np.ndarray:
        """Returns timestamps of events with name ``event_name``"""
        event_id = self.event_names.index(event_name)

        # Raise Error if not found:
        if event_id == -1:
            raise ValueError(
                f"Event {event_name} not found in Events {self.event_names}"
            )

        timestamps = self.events[event_id].times.magnitude

        # Transform from unit seconds to timestamps using the sampling frequency:
        timestamps *= self.fs

        return np.rint(timestamps).astype(int)

    def t(self):
        return np.linspace(
            0, (self.data.shape[-1] - 1.0) / self.fs, self.data.shape[-1]
        )

    def free(self):
        if hasattr(self, "data"):
            del self.data

    def __getitem__(self, key):
        return self.data[self.ch_dict[key]]
