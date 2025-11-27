# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from loguru import logger

from lib.data import SmrImporter, highpass, bandpass
from lib.epochs import Epochs

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman"}
)

# %%

DATA_DIR = Path("../Pilot1 - Tao Liu 20251118")
DATA_FILE = DATA_DIR / "[NBM-TUS] Tao Task.smr"

DATA_DIR = Path("../Pilot2 - Faissal Sharif 20251119")
DATA_FILE = DATA_DIR / "[NBM-TUS] Faissal Posner Main.smr"

DATA_DIR = Path("../Pilot3 - Bo Yin 20251120")
DATA_FILE = DATA_DIR / "[NBM-TUS] Bo Posner Main.smr"

DATA_DIR = Path("../Pilot5 - PD Patient 20251212")
DATA_FILE = DATA_DIR / "[NBM-TUS] Posner-Main.smr"

DATA_DIR.exists()

# %%
importer = SmrImporter(DATA_FILE)
n_channels = len(importer.ch_names)

eeg_chans = [
    "FP1",
    "FP2",
    "F3",
    "Fz",
    "F4",
    "C3",
    "Cz",
    "C4",
    "P3",
    "Pz",
    "P4",
    "O1",
    "Oz",
    "O2",
]

eeg_chan_pos = {
    "FP1": (0, 0),
    "FP2": (0, 2),
    "F3": (1, 0),
    "Fz": (1, 1),
    "F4": (1, 2),
    "C3": (2, 0),
    "Cz": (2, 1),
    "C4": (2, 2),
    "P3": (3, 0),
    "Pz": (3, 1),
    "P4": (3, 2),
    "O1": (4, 0),
    "Oz": (4, 1),
    "O2": (4, 2),
}
eeg_chan_inds = [importer.ch_names.index(ch) for ch in eeg_chans]
n_channels = len(eeg_chan_inds)

eeg_data = importer.data[eeg_chan_inds, :]


# %%
trigger_chans = ["FIO", "DAC"]
trigger_chan_inds = [importer.ch_names.index(ch) for ch in trigger_chans]
trigger_data = importer.data[trigger_chan_inds, :]

other_chans = ["ECG", "AccX", "AccY", "AccZ"]
other_chan_inds = [importer.ch_names.index(ch) for ch in other_chans]
other_data = importer.data[other_chan_inds, :]

logger.info(
    f"eeg: {eeg_data.shape}, trigger: {trigger_data.shape}, other: {other_data.shape}"
)


# %%
# Run pre-preocessing on longitudinal data

# Re-ferencing

# Filtering (highpass, notch...)
eeg_data = highpass(eeg_data, fs=float(importer.fs), cutoff_freq=0.5)
# eeg_data = bandpass(eeg_data, fs=float(importer.fs), band_freq=(8, 12))
other_data = highpass(other_data, fs=float(importer.fs), cutoff_freq=0.5)


# %% Epoch data
timestamps = importer.get_event_timestamps("TrlSt")

# %%
EPOCH_LIMS = (-3, 5)


def epoch_data(data, timestamps, fs, epoch_lims: tuple[float, float] = (-3, 5)):
    epoch_lims_samp = (int(fs * epoch_lims[0]), int(fs * epoch_lims[1]))
    print(f"Epoch limits: {epoch_lims_samp}")

    pre = int(round(fs * abs(epoch_lims[0])))
    post = int(round(fs * epoch_lims[1]))
    epoch_len = pre + post

    epochs = np.zeros((len(timestamps), data.shape[0], epoch_len)) * np.nan

    for ix, timestamp in enumerate(timestamps):
        start_data = timestamp + epoch_lims_samp[0]
        end_data = timestamp + epoch_lims_samp[1]

        pad_left = max(0, -start_data)
        pad_right = max(0, end_data - data.shape[1])
        start_data = max(0, start_data)
        end_data = min(data.shape[1], end_data)

        print(f"Epoch {ix}: t={timestamp}, start={start_data}, end={end_data}")
        print(f"Padding: {pad_left} left, {pad_right} right")

        epochs[ix, :, pad_left : epoch_len - pad_right] = data[:, start_data:end_data]

        if pad_left:
            print(f"Padding left epoch {ix} with {pad_left} samples")
            epochs[ix, :, :pad_left] = np.flip(
                data[:, start_data : start_data + pad_left], axis=-1
            )

        if pad_right:
            print(f"Padding right epoch {ix} with {pad_right} samples")
            epochs[ix, :, epoch_len - pad_right :] = np.flip(
                data[:, end_data - pad_right : end_data], axis=-1
            )

    assert not np.any(np.isnan(epochs)), f"Epochs contain NaNs"
    t_epoch = np.linspace(0, (epoch_len - 1) / fs, epoch_len) + epoch_lims[0]
    return epochs, t_epoch


eeg_epochs, t_epoch = epoch_data(eeg_data, timestamps, importer.fs, EPOCH_LIMS)
trigger_epochs, t_epoch = epoch_data(trigger_data, timestamps, importer.fs, EPOCH_LIMS)
other_epochs, t_epoch = epoch_data(other_data, timestamps, importer.fs, EPOCH_LIMS)

# %%
logger.info(f"Epochs: {eeg_epochs.shape}, {trigger_epochs.shape}, {other_epochs.shape}")

# %%

plot_data = trigger_epochs

fig, ax = plt.subplots(plot_data.shape[1], 1, figsize=(8, 4), sharex=True)

for ch in range(plot_data.shape[1]):
    ax[ch].plot(t_epoch, plot_data[:, ch, :].T, alpha=0.01)
    ax[ch].axvline(x=0, color="k", linestyle="--")

fig.tight_layout()

# %%
dac_vals = trigger_epochs[:, 1, :].flatten()

hist, bins = np.histogram(dac_vals, bins=1048, density=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(bins[:-1], hist, alpha=0.5)

from scipy.signal import find_peaks

peaks, _ = find_peaks(hist, height=0.5e-5)
bins[peaks]

ax.scatter(bins[peaks], hist[peaks], c="r")

# %%

level_names = [
    "fixation_cross",
    "left_arrow",
    "right_arrow",
    "both_arrows",
    "left_target",
    "right_target",
    "left_button_press",
    "right_button_press",
]

levels_parser = dict()
levels_parser["fixation_cross"] = "fixation_cross"
levels_parser["left_arrow"] = "left"
levels_parser["right_arrow"] = "right"
levels_parser["both_arrows"] = "both"
levels_parser["left_target"] = "left"
levels_parser["right_target"] = "right"
levels_parser["left_button_press"] = "left"
levels_parser["right_button_press"] = "right"

levels = dict()
for kk, level_name in enumerate(level_names):
    levels[level_name] = bins[peaks[kk]]


# %%
def find_threshold_crossings(
    signal: np.ndarray, threshold: float, skip: int
) -> list[int]:
    """
    Return indices where ``signal`` first crosses ``threshold`` and skip ``skip`` samples before searching again.
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be 1-D")
    if skip < 1:
        raise ValueError("Skip must be >= 1")

    hits: list[int] = []
    cursor = 0
    total = signal.shape[0]

    while cursor < total:
        above = np.flatnonzero(signal[cursor:] >= threshold)
        if above.size == 0:
            break
        idx = cursor + int(above[0])
        hits.append(idx)
        cursor = idx + skip

    return hits


def match_val_to_levels(
    val: float, levels: dict[str, float], tolerance: float = 25_000
) -> str | None:
    """
    Return the name of the closest ``levels`` entry to ``val`` if it is within ``tolerance``.
    """
    if not levels:
        raise ValueError("levels mapping must not be empty")

    names, values = zip(*levels.items())
    values_arr = np.asarray(values, dtype=float)
    diffs = np.abs(values_arr - float(val))
    best_idx = int(np.argmin(diffs))

    if diffs[best_idx] <= tolerance:
        return names[best_idx]

    return None


# %%
roi_ix = (np.array((-0.1, 3)) + 3.0).astype(int) * importer.fs
n_epochs = trigger_epochs.shape[0]

rts = []
hits_cont = []
events_cont = []
for epoch in range(n_epochs):
    fio_chan = trigger_epochs[epoch, 0, int(roi_ix[0]) : int(roi_ix[1])]
    hits = find_threshold_crossings(np.diff(fio_chan), 100_000, 512)[:4]
    hits = [int(round(hit + roi_ix[0])) for hit in hits]
    hits_cont.append(hits)
    assert len(hits) == 4, f"Epoch {epoch} does not have 4 hits: {hits}"

    level_log = []
    for hit in hits:
        level = match_val_to_levels(
            trigger_epochs[
                epoch, 1, hit + int(4096 * 0.075) : hit + int(4096 * 0.125)
            ].mean(),
            levels,
        )
        level_log.append(levels_parser[level])
    events_cont.append(level_log)
    print(f"Epoch {epoch}: {hits}, levels={level_log}")

epochs = Epochs(
    eeg_data=eeg_epochs,
    eeg_names=eeg_chans,
    trigger_data=trigger_epochs,
    trigger_ch_names=trigger_chans,
    other_data=other_epochs,
    other_ch_names=other_chans,
    fs=float(importer.fs),
    event_timestamps=np.array(hits_cont),
    events=np.array(events_cont),
    epoch_lims=EPOCH_LIMS,
)


# %%

from mne.time_frequency import tfr_array_morlet


power = tfr_array_morlet(
    epochs.eeg_data,
    freqs=np.arange(2.0, 120, 0.5),
    n_cycles=3,
    sfreq=float(importer.fs),
    n_jobs=2,
    output="power",
)

fig, ax = plt.subplots(5, 3, figsize=(13, 13), sharex=True, sharey=True)

for ch_ix, eeg_chan in enumerate(eeg_chans):
    ax_ = ax[eeg_chan_pos[eeg_chan][0], eeg_chan_pos[eeg_chan][1]]

    im = ax_.imshow(
        power[:, ch_ix, :, :].mean(axis=0),
        aspect="auto",
        origin="lower",
        vmin=-0.5,
        vmax=0.5,
        cmap="RdBu_r",
    )
    ax_.set_title(eeg_chan)
    fig.colorbar(im, ax=ax_)
    # for event_timestamp in epochs.query("incongruent-all").event_timestamps:
    ax_.axvline(x=0, color="k", linestyle="--")
    ax_.axvline(x=epochs.t[epochs.event_timestamps[0, 1]])
    ax_.axvline(x=epochs.t[epochs.event_timestamps[0, 2]])
    ax_.axvline(x=epochs.t[epochs.event_timestamps[0, 3]])

    ax_.set_xlim((-1.5, 2.5))
    ax_.set_ylim(-5, 5)


# %% RT Analysis

from scipy.stats import ttest_ind
import seaborn as sns

conds = ["congruent-all", "incongruent-all", "neutral-all"]

epochs_cond = dict()
for cond in conds:
    epochs_cond[cond] = epochs.query(cond)

fig, ax = plt.subplots(1, 1, figsize=(7.9, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for ix, cond in enumerate(conds):
    epochs_ = epochs_cond[cond]
    print(epochs_.rts.max())
    sns.swarmplot(x=np.zeros(epochs_.n_epochs) + ix, y=epochs_.rts, alpha=0.5, ax=ax)
    ax.boxplot(
        epochs_.rts,
        positions=[ix],
        widths=0.4,
        notch=False,
        showfliers=False,
        medianprops=dict(color="k", linestyle="--"),
    )

ax.set_ylim(0.1, 0.7)
ax.set_ylabel("Reaction Time [s]")

ax.set_xticklabels(
    [
        f"Congruent\n(n={epochs_cond['congruent-all'].n_epochs})",
        f"Incongruent\n(n={epochs_cond['incongruent-all'].n_epochs})",
        f"Neutral\n(n={epochs_cond['neutral-all'].n_epochs})",
    ]
)

fig.tight_layout()
fig.savefig("outputs/congruent-incongruent-neutral-rts.svg")
# Pairwise unpaired t-tests

# print(ttest_ind(epochs_cond["congruent-left"].rts, epochs_cond["congruent-right"].rts))
print(ttest_ind(epochs_cond["congruent-all"].rts, epochs_cond["incongruent-all"].rts))
print(ttest_ind(epochs_cond["congruent-all"].rts, epochs_cond["neutral-all"].rts))
print(ttest_ind(epochs_cond["incongruent-all"].rts, epochs_cond["neutral-all"].rts))

# %%
conds = [
    "congruent-left",
    "congruent-right",
    "incongruent-left",
    "incongruent-right",
    "neutral-left",
    "neutral-right",
    "mistakes",
]

epochs_cond = dict()
for cond in conds:
    epochs_cond[cond] = epochs.query(cond)

fig, ax = plt.subplots(1, 1, figsize=(7.9, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for ix, cond in enumerate(conds):
    epochs_ = epochs_cond[cond]
    if epochs_.eeg_data.size == 0:
        continue
    print(epochs_.rts.max())
    sns.swarmplot(x=np.zeros(epochs_.n_epochs) + ix, y=epochs_.rts, alpha=0.5, ax=ax)
    ax.boxplot(
        epochs_.rts,
        positions=[ix],
        widths=0.4,
        notch=False,
        showfliers=False,
        medianprops=dict(color="k", linestyle="--"),
        showmeans=False,
    )

ax.set_ylim(0.1, 0.7)
ax.set_ylabel("Reaction Time [s]")

ax.set_xticks([0, 1, 2, 3, 4, 5, 6])

ax.set_xticklabels([f"{cond}\n(n={epochs_cond[cond].n_epochs})" for cond in conds])

fig.tight_layout()
fig.savefig("outputs/detailed-rts.svg")


# %%
# Start looking at E-Phys

fig, ax = plt.subplots(5, 3, figsize=(13, 13), sharex=True, sharey=True)

for ch_ix, eeg_chan in enumerate(eeg_chans):
    ax_ = ax[eeg_chan_pos[eeg_chan][0], eeg_chan_pos[eeg_chan][1]]
    ax_.plot(
        epochs.t,
        epochs.query("congruent-left").eeg_data[:, ch_ix, :].T,
        c="gray",
        alpha=0.01,
    )

    ax_.plot(
        epochs.t,
        epochs.query("congruent-left").eeg_data[:, ch_ix, :].mean(axis=0),
        c="C2",
        alpha=0.9,
    )
    ax_.set_title(eeg_chan)
    # for event_timestamp in epochs.query("incongruent-all").event_timestamps:
    ax_.axvline(x=0, color="k", linestyle="--")
    ax_.axvline(x=epochs.t[epochs.event_timestamps[0, 1]])
    ax_.axvline(x=epochs.t[epochs.event_timestamps[0, 2]])
    ax_.axvline(x=epochs.t[epochs.event_timestamps[0, 3]])

    ax_.set_xlim((-1.5, 2.5))
    ax_.set_ylim(-5, 5)

fig.tight_layout()

# %%

epochs.eeg_data.shape

# %%
hits = find_threshold_crossings(np.diff(fio_chan[11, :]), 100_000, 512)[:4]
hits_global = [int(round(hit + roi_ix[0])) for hit in hits]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t_epoch, trigger_epochs[11, 1, :], alpha=1)
for hit in hits_global:
    ax.axvline(x=t_epoch[hit], color="r", linestyle="--")

# %%
val = trigger_epochs[
    11, 1, hits_global[0] + int(4096 * 0.1) : hits_global[0] + int(4096 * 0.1) * 3
].mean()


# %%

# %%
reaction_time = t_epoch[hits_global[3]] - t_epoch[hits_global[2]]
reaction_time

# %%
rts_other = np.array(
    [
        0.3744174583116550,
        0.340773125004489,
        0.38551712495973300,
        0.302270625019446,
        0.34534708329010800,
        0.3681636250112210,
        0.3945229999953880,
        0.37611062498763200,
        0.46283941669389600,
        0.38555779168382300,
        0.4015147083555350,
        0.3660478750243780,
        0.312130291655194,
        0.3295619583223020,
        0.34525754168862500,
        0.3578679166967050,
        0.2817292083054780,
        0.36173475004034100,
        0.4257311249966730,
        0.42912225000327500,
        0.29709800001001000,
        0.3220441250014120,
    ]
)

# %%
rts = np.array(rts)[: len(rts_other)]

print(rts - rts_other)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.scatter(rts[: len(rts_other)], rts_other, alpha=0.1)

# %%
