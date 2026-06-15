# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
from loguru import logger
import mne

from lib.data import SmrImporter, highpass


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Math"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Latin Modern Math",
    "mathtext.it": "Latin Modern Math:italic",
    "mathtext.bf": "Latin Modern Math:bold",
})


# %%

root = Path("/Volumes/T7/Projects/NBM-TUS/EEG/sub-3001")

fname = root / "Day 1" / "sub3001-PreTUS-Rest.smrx"
fname = root / "Day 1" / "sub3001-PostTUS-Rest.smrx"

fname = root / "Day 2" / "sub3001-PreTUS-Day2-Rest.smrx"
fname = root / "Day 2" / "sub3001-day2-postTUS-Rest.smrx"

# fname = root / "Day 3" / "sub3001-Day3-PreTUS-Rest.smrx"
# fname = root / "Day 3" / "sub3001-day3-PostTUS-Rest.smrx"

# %%
logger.info(f"Loading data from {fname.name}...")
importer = SmrImporter(fname)

eeg_chans = [
    "FP1", "FP2", "F3", "Fz", "F4", "FCz", "C3", "Cz", "C4",
    "CP3", "CPz", "CP4", "P3", "Pz", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2"
]

# Robust channel selection: only extract channels that are present in the file
present_chans = [ch for ch in eeg_chans if ch in importer.ch_names]
present_indices = [importer.ch_names.index(ch) for ch in present_chans]

if len(present_indices) > 0:
    eeg_data = importer.data[present_indices, :]
    logger.info(f"Applying 0.5 Hz highpass filter to {len(present_chans)} channels...")
    eeg_data = highpass(eeg_data, fs=float(importer.fs), cutoff_freq=0.5)
else:
    logger.warning("No target EEG channels found in importer data!")
    eeg_data = np.empty((0, importer.data.shape[1]))

# %%
# Slice into Eyes Open (EO) and Eyes Closed (EC) epochs
t_eo = importer.get_event_timestamps("EO")
t_ec = importer.get_event_timestamps("EC")

def extract_epoch_data(data, timestamps, fallback_end=None):
    n = len(timestamps)
    if n == 2:
        logger.info(f"Extracting epoch from marker 0 ({timestamps[0]}) to marker 1 ({timestamps[1]})")
        return data[:, timestamps[0]:timestamps[1]]
    elif n == 4:
        logger.info(f"Extracting concatenated epoch from marker 0:1 ({timestamps[0]}:{timestamps[1]}) and marker 2:3 ({timestamps[2]}:{timestamps[3]})")
        part1 = data[:, timestamps[0]:timestamps[1]]
        part2 = data[:, timestamps[2]:timestamps[3]]
        return np.concatenate([part1, part2], axis=-1)
    elif n == 1:
        if fallback_end is not None:
            logger.info(f"Extracting epoch from marker 0 ({timestamps[0]}) to fallback end ({fallback_end})")
            return data[:, timestamps[0]:fallback_end]
        else:
            logger.info(f"Extracting epoch from marker 0 ({timestamps[0]}) to end of data")
            return data[:, timestamps[0]:]
    else:
        raise ValueError(f"Unexpected number of markers: {n}. Expected 1, 2, or 4.")

# Slicing data based on parsed markers
eo_data = extract_epoch_data(eeg_data, t_eo, fallback_end=t_ec[0] if len(t_ec) > 0 else None)
ec_data = extract_epoch_data(eeg_data, t_ec, fallback_end=None)

logger.info(f"Eyes Open data length: {eo_data.shape[1] / importer.fs:.2f} seconds")
logger.info(f"Eyes Closed data length: {ec_data.shape[1] / importer.fs:.2f} seconds")

# Compute PSD using Welch's method (4s window)
nperseg = int(4 * importer.fs)
f_eo, psd_eo = welch(eo_data, fs=importer.fs, nperseg=nperseg)
f_ec, psd_ec = welch(ec_data, fs=importer.fs, nperseg=nperseg)

# Compute average alpha power (8-12 Hz)
alpha_mask = (f_eo >= 6) & (f_eo <= 10)
if len(present_chans) > 0:
    alpha_eo = np.mean(psd_eo[:, alpha_mask], axis=1)
    alpha_ec = np.mean(psd_ec[:, alpha_mask], axis=1)
    # Alpha reactivity calculations
    with np.errstate(divide='ignore', invalid='ignore'):
        reactivity_ratio = np.log(np.where(alpha_ec > 0, alpha_ec, np.nan)) - np.log(np.where(alpha_eo > 0, alpha_eo, np.nan))
        reactivity_db = 10 * np.log10(np.where(alpha_ec > 0, alpha_ec, np.nan) / np.where(alpha_eo > 0, alpha_eo, np.nan))
else:
    alpha_eo = np.array([])
    alpha_ec = np.array([])
    reactivity_ratio = np.array([])
    reactivity_db = np.array([])

# Reconstruct metrics for all target channels (filling with NaN for missing ones)
alpha_eo_full = np.full(len(eeg_chans), np.nan)
alpha_ec_full = np.full(len(eeg_chans), np.nan)
reactivity_ratio_full = np.full(len(eeg_chans), np.nan)
reactivity_db_full = np.full(len(eeg_chans), np.nan)

# Map PSD arrays for plotting
psd_eo_full = np.full((len(eeg_chans), len(f_eo)), np.nan)
psd_ec_full = np.full((len(eeg_chans), len(f_ec)), np.nan)

for i, chan in enumerate(present_chans):
    idx_full = eeg_chans.index(chan)
    alpha_eo_full[idx_full] = alpha_eo[i]
    alpha_ec_full[idx_full] = alpha_ec[i]
    reactivity_ratio_full[idx_full] = reactivity_ratio[i]
    reactivity_db_full[idx_full] = reactivity_db[i]
    psd_eo_full[idx_full, :] = psd_eo[i, :]
    psd_ec_full[idx_full, :] = psd_ec[i, :]

# Create a summary DataFrame
df_summary = pd.DataFrame({
    "Channel": eeg_chans,
    "Alpha_EO_Power": alpha_eo_full,
    "Alpha_EC_Power": alpha_ec_full,
    "Reactivity_Ratio": reactivity_ratio_full,
    "Reactivity_dB": reactivity_db_full
})

logger.info("\n" + df_summary.to_string(index=False))

# Plot PSD for posterior channels (P*, PO*, O*) if present
candidate_chans = ["P3", "Pz", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2"]
present_candidates = [ch for ch in candidate_chans if ch in present_chans]

if len(present_candidates) > 0:
    ncols = min(3, len(present_candidates))
    nrows = (len(present_candidates) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=True, squeeze=False)
    axes = axes.flatten()
    
    for i, chan in enumerate(present_candidates):
        ax = axes[i]
        ax.set_title(f"Channel {chan}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"Power Spectral Density ($\mu V^2/Hz$)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim([0, 20])
        
        ch_idx = eeg_chans.index(chan)
        # Plot EO vs EC in the alpha range (1-30 Hz for visualization)
        vis_mask = (f_eo >= 1) & (f_eo <= 30)
        ax.plot(f_eo[vis_mask], psd_eo_full[ch_idx, vis_mask], label="Eyes Open (EO)", color="#3182bd", linewidth=2)
        ax.plot(f_ec[vis_mask], psd_ec_full[ch_idx, vis_mask], label="Eyes Closed (EC)", color="#de2d26", linewidth=2)
        ax.legend()
        
    # Hide unused axes
    for j in range(len(present_candidates), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Resting State Alpha Reactivity - Posterior Channels", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "alpha_reactivity.svg"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved alpha reactivity plot to {plot_path}")
else:
    logger.warning("None of the posterior channels are present in the data. Skipping plot.")

# %%
# Topoplots of Alpha Activity and Reactivity
# Filter out channels that are present but flat/zero (e.g. power == 0) to avoid log(0) issues
present_eeg_chans = [
    ch for i, ch in enumerate(present_chans)
    if ch in eeg_chans and alpha_eo[i] > 0 and alpha_ec[i] > 0
]
if len(present_eeg_chans) > 0:
    logger.info("Generating topoplots for alpha activity...")
    # Extract data for present channels matching the indices
    present_indices_in_full = [eeg_chans.index(ch) for ch in present_eeg_chans]
    
    alpha_eo_plot = np.log(alpha_eo_full[present_indices_in_full])
    alpha_ec_plot = np.log(alpha_ec_full[present_indices_in_full])
    reactivity_plot = reactivity_ratio_full[present_indices_in_full]
    
    # Create Info object and set standard 10-20 montage (case-insensitive)
    info_topo = mne.create_info(ch_names=present_eeg_chans, sfreq=importer.fs, ch_types="eeg")
    info_topo.set_montage("standard_1020", match_case=False)
    
    fig_topo, axes_topo = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common scaling for EO and EC
    vmin = min(np.nanmin(alpha_eo_plot), np.nanmin(alpha_ec_plot))
    vmax = max(np.nanmax(alpha_eo_plot), np.nanmax(alpha_ec_plot))
    
    # 1. Eyes Open Alpha Power
    axes_topo[0].set_title("Alpha Power (log) - Eyes Open (EO)")
    im_eo = mne.viz.plot_topomap(alpha_eo_plot, info_topo, axes=axes_topo[0], vlim=(vmin, vmax), cmap="viridis", show=False)[0]
    fig_topo.colorbar(im_eo, ax=axes_topo[0], orientation="vertical", shrink=0.7, label=r"log Power ($\log(\mu V^2/Hz)$)")
    
    # 2. Eyes Closed Alpha Power
    axes_topo[1].set_title("Alpha Power (log) - Eyes Closed (EC)")
    im_ec = mne.viz.plot_topomap(alpha_ec_plot, info_topo, axes=axes_topo[1], vlim=(vmin, vmax), cmap="viridis", show=False)[0]
    fig_topo.colorbar(im_ec, ax=axes_topo[1], orientation="vertical", shrink=0.7, label=r"log Power ($\log(\mu V^2/Hz)$)")
    
    # 3. Reactivity Ratio (log EC - log EO)
    axes_topo[2].set_title("Alpha Reactivity\n(log EC - log EO)")
    # Center colormap at 0 for reactivity
    v_abs = max(abs(np.nanmin(reactivity_plot)), abs(np.nanmax(reactivity_plot))) if len(reactivity_plot) > 0 else 1.0
    im_react = mne.viz.plot_topomap(reactivity_plot, info_topo, axes=axes_topo[2], vlim=(-v_abs, v_abs), cmap="RdBu_r", show=False)[0]
    fig_topo.colorbar(im_react, ax=axes_topo[2], orientation="vertical", shrink=0.7, label="log Difference")
    
    plt.suptitle("Resting State Alpha Activity & Reactivity Topoplots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    topo_plot_path = output_dir / "alpha_reactivity_topomap.svg"
    plt.savefig(topo_plot_path, dpi=300)
    logger.info(f"Saved alpha activity topoplots to {topo_plot_path}")
else:
    logger.warning("No present EEG channels found to generate topoplots.")

# %%
