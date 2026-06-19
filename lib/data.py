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


class MockTimes:
    def __init__(self, magnitude):
        self.magnitude = np.array(magnitude)


class MockEvent:
    def __init__(self, name, times_in_seconds):
        self.name = name
        self.times = MockTimes(times_in_seconds)


class SmrImporter:
    """Reads signals and metadata (sampling frequency ``fs``, channel names ``ch_names``) from SMR/SMRX file"""

    def __init__(self, fname: "os.PathLike | str"):
        fname_str = str(fname)
        if fname_str.lower().endswith(".smrx"):
            import subprocess
            import pickle
            from pathlib import Path

            try:
                lib_dir = Path(__file__).parent
                res_path = subprocess.run(
                    ["pixi", "run", "-e", "sonpy", "python", "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=lib_dir.parent
                )
                sonpy_python = Path(res_path.stdout.strip())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to dynamically query pixi for the 'sonpy' python executable: {e}\n"
                    "Make sure pixi is installed and you have run 'pixi install --all'."
                )

            if not sonpy_python.exists():
                raise FileNotFoundError(
                    f"Could not find the 'sonpy' environment python binary at: {sonpy_python}"
                )

            # Subprocess python script to read SMRX using sonpy in the 3.9 environment
            script = """
import sys
import pickle
import numpy as np
from sonpy import lib as sp

fname_str = sys.argv[1]
f = sp.SonFile(fname_str, True)
if f.GetOpenError() != 0:
    sys.exit(f"Error opening file: {sp.GetErrorString(f.GetOpenError())}")

# Find active analog channels
analog_chans = []
for chan in range(f.MaxChannels()):
    t = f.ChannelType(chan)
    if t in (sp.DataType.Adc, sp.DataType.RealWave):
        analog_chans.append(chan)

if not analog_chans:
    sys.exit("No active analog channels found")

ref_chan = analog_chans[0]
timebase = f.GetTimeBase()
ref_divide = f.ChannelDivide(ref_chan)
fs = 1.0 / (timebase * ref_divide)

max_time = f.ChannelMaxTime(ref_chan)
n_points = int(max_time / ref_divide)

ch_names = []
data_list = []
for chan in analog_chans:
    ch_names.append(f.GetChannelTitle(chan))
    data_list.append(f.ReadFloats(chan, n_points, 0))

data = np.array(data_list, dtype=np.float32)

# Read event channels
events = []
for chan in range(f.MaxChannels()):
    t = f.ChannelType(chan)
    if t in (sp.DataType.EventFall, sp.DataType.EventRise, sp.DataType.EventBoth):
        name = f.GetChannelTitle(chan)
        max_t = f.ChannelMaxTime(chan)
        if max_t >= 0:
            ticks = f.ReadEvents(chan, 100000, 0, max_t + 1)
            times_sec = np.array(ticks, dtype=np.float64) * timebase
        else:
            times_sec = np.array([], dtype=np.float64)
        events.append({"name": name, "times": times_sec})

result = {
    "fs": fs,
    "ch_names": ch_names,
    "data": data,
    "events": events,
}

sys.stdout.buffer.write(pickle.dumps(result))
"""
            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            res = subprocess.run(
                [str(sonpy_python), "-c", script, fname_str],
                capture_output=True,
                env=env
            )

            if res.returncode != 0:
                raise RuntimeError(
                    f"Failed to read SMRX file using sonpy environment subprocess:\n"
                    f"{res.stderr.decode('utf-8')}"
                )

            result = pickle.loads(res.stdout)

            self.fs = result["fs"]
            self.ch_names = result["ch_names"]
            self.data = result["data"]
            self.ch_dict = {name: i for i, name in enumerate(self.ch_names)}

            self.events = []
            for ev in result["events"]:
                self.events.append(MockEvent(ev["name"], ev["times"]))

            self.event_names = [event.name for event in self.events]
            print(self.event_names)
            print(self.ch_names)

        else:
            reader = Spike2IO(filename=fname_str)
            block = reader.read()[0]
            segment = block.segments[0]
            analog_signal = segment.analogsignals[0]
            self.events = segment.events
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
