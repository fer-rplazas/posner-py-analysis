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

root_dir = Path("/Volumes/T7/Projects/NBM-TUS/EEG/sub-3001")

# Target EEG channels
eeg_chans = [
    "FP1", "FP2", "F3", "Fz", "F4", "FCz", "C3", "Cz", "C4",
    "CP3", "CPz", "CP4", "P3", "Pz", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2"
]

# Define available days and their relative paths
day_files = {
    1: {
        "pre": "Day 1/sub3001-PreTUS-Rest.smrx",
        "post": "Day 1/sub3001-PostTUS-Rest.smrx"
    },
    2: {
        "pre": "Day 2/sub3001-PreTUS-Day2-Rest.smrx",
        "post": "Day 2/sub3001-day2-postTUS-Rest.smrx"
    },
    3: {
        "pre": "Day 3/sub3001-Day3-PreTUS-Rest.smrx",
        "post": "Day 3/sub3001-day3-PostTUS-Rest.smrx"
    }
}

# %%

def extract_epoch_data(data, timestamps, fallback_end=None):
    n = len(timestamps)
    if n == 2:
        return data[:, timestamps[0]:timestamps[1]]
    elif n == 4:
        part1 = data[:, timestamps[0]:timestamps[1]]
        part2 = data[:, timestamps[2]:timestamps[3]]
        return np.concatenate([part1, part2], axis=-1)
    elif n == 1:
        if fallback_end is not None:
            return data[:, timestamps[0]:fallback_end]
        else:
            return data[:, timestamps[0]:]
    else:
        raise ValueError(f"Unexpected number of markers: {n}. Expected 1, 2, or 4.")

def analyze_file(fname):
    logger.info(f"Loading data from {fname.name}...")
    importer = SmrImporter(fname)
    
    present_chans = [ch for ch in eeg_chans if ch in importer.ch_names]
    present_indices = [importer.ch_names.index(ch) for ch in present_chans]
    
    if len(present_indices) == 0:
        raise ValueError(f"No target EEG channels found in {fname.name}")
        
    eeg_data = importer.data[present_indices, :]
    eeg_data = highpass(eeg_data, fs=float(importer.fs), cutoff_freq=0.5)
    
    t_eo = importer.get_event_timestamps("EO")
    t_ec = importer.get_event_timestamps("EC")
    
    eo_data = extract_epoch_data(eeg_data, t_eo, fallback_end=t_ec[0] if len(t_ec) > 0 else None)
    ec_data = extract_epoch_data(eeg_data, t_ec, fallback_end=None)
    
    # Compute PSD using Welch's method (4s window)
    nperseg = int(4 * importer.fs)
    f_eo, psd_eo = welch(eo_data, fs=importer.fs, nperseg=nperseg)
    f_ec, psd_ec = welch(ec_data, fs=importer.fs, nperseg=nperseg)
    
    # Compute average alpha power (6-10 Hz)
    alpha_mask = (f_eo >= 6) & (f_eo <= 10)
    alpha_eo = np.mean(psd_eo[:, alpha_mask], axis=1)
    alpha_ec = np.mean(psd_ec[:, alpha_mask], axis=1)
    
    # Alpha reactivity calculations (log difference)
    with np.errstate(divide='ignore', invalid='ignore'):
        reactivity = np.log(np.where(alpha_ec > 0, alpha_ec, np.nan)) - np.log(np.where(alpha_eo > 0, alpha_eo, np.nan))
        
    # Reconstruct metrics for all target channels (filling with NaN for missing ones)
    alpha_eo_full = np.full(len(eeg_chans), np.nan)
    alpha_ec_full = np.full(len(eeg_chans), np.nan)
    reactivity_full = np.full(len(eeg_chans), np.nan)
    
    for i, chan in enumerate(present_chans):
        idx_full = eeg_chans.index(chan)
        alpha_eo_full[idx_full] = alpha_eo[i]
        alpha_ec_full[idx_full] = alpha_ec[i]
        reactivity_full[idx_full] = reactivity[i]
        
    return alpha_eo_full, alpha_ec_full, reactivity_full

# %%

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Loop over each day to compare PreTUS vs PostTUS
for day, files in day_files.items():
    pre_path = root_dir / files["pre"]
    post_path = root_dir / files["post"]
    
    if not pre_path.exists() or not post_path.exists():
        logger.warning(f"Files for Day {day} not found. Skipping.")
        continue
        
    logger.info(f"\n========================================\nAnalyzing Day {day}...\n========================================")
    
    try:
        pre_eo, pre_ec, pre_react = analyze_file(pre_path)
        post_eo, post_ec, post_react = analyze_file(post_path)
        
        # Calculate percentage changes
        with np.errstate(divide='ignore', invalid='ignore'):
            eo_change = (post_eo - pre_eo) / pre_eo * 100
            ec_change = (post_ec - pre_ec) / pre_ec * 100
            react_change = (post_react - pre_react) / np.abs(pre_react) * 100
            
        # Create comparison DataFrame
        df_compare = pd.DataFrame({
            "Channel": eeg_chans,
            "Pre_EO": pre_eo,
            "Post_EO": post_eo,
            "EO_Pct_Change": eo_change,
            "Pre_EC": pre_ec,
            "Post_EC": post_ec,
            "EC_Pct_Change": ec_change,
            "Pre_Reactivity": pre_react,
            "Post_Reactivity": post_react,
            "Reactivity_Pct_Change": react_change
        })
        
        # Save CSV comparison table
        csv_path = output_dir / f"day{day}_pre_post_comparison.csv"
        df_compare.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison CSV to {csv_path}")
        
        # Filter out NaN rows to display valid channel comparisons in the log
        df_display = df_compare.dropna(subset=["Pre_EO", "Post_EO"])
        logger.info(f"\nDay {day} Comparison Table:\n" + df_display.to_string(index=False))
        
        # --- Plot 1: Bar Chart of Percentage Changes ---
        valid_mask = ~np.isnan(eo_change)
        plot_chans = [eeg_chans[i] for i in range(len(eeg_chans)) if valid_mask[i]]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(plot_chans))
        width = 0.25
        
        ax.bar(x - width, eo_change[valid_mask], width, label="EO Alpha Power % Change", color="#3182bd")
        ax.bar(x, ec_change[valid_mask], width, label="EC Alpha Power % Change", color="#de2d26")
        ax.bar(x + width, react_change[valid_mask], width, label="Reactivity % Change", color="#31a354")
        
        ax.set_xlabel("Channel")
        ax.set_ylabel("Percentage Change (%)")
        ax.set_title(f"EEG Alpha Metrics Percentage Change (Post-TUS vs. Pre-TUS) - Day {day}")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_chans)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plot_path = output_dir / f"day{day}_pre_post_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved comparison bar plot to {plot_path}")
        
        # --- Plot 2: Topoplots of Percentage Changes ---
        # Select channels that are present and not flat (power > 0)
        valid_topo_mask = (
            ~np.isnan(pre_eo) & ~np.isnan(post_eo) & 
            (pre_eo > 0) & (post_eo > 0) & 
            (pre_ec > 0) & (post_ec > 0)
        )
        present_topo_chans = [eeg_chans[i] for i in range(len(eeg_chans)) if valid_topo_mask[i]]
        
        if len(present_topo_chans) > 0:
            info_topo = mne.create_info(ch_names=present_topo_chans, sfreq=1000.0, ch_types="eeg")
            info_topo.set_montage("standard_1020", match_case=False)
            
            fig_topo, axes_topo = plt.subplots(1, 3, figsize=(15, 5))
            
            eo_pct_plot = eo_change[valid_topo_mask]
            ec_pct_plot = ec_change[valid_topo_mask]
            react_pct_plot = react_change[valid_topo_mask]
            
            # Common color scaling for power changes
            v_power_abs = max(
                abs(np.nanmin(eo_pct_plot)), abs(np.nanmax(eo_pct_plot)),
                abs(np.nanmin(ec_pct_plot)), abs(np.nanmax(ec_pct_plot))
            ) if len(eo_pct_plot) > 0 else 10.0
            
            # Divergent colormaps centered at 0 are perfect for percentage change
            # 1. EO Change Topomap
            axes_topo[0].set_title("EO Alpha Power % Change")
            im_eo = mne.viz.plot_topomap(eo_pct_plot, info_topo, axes=axes_topo[0], vlim=(-v_power_abs, v_power_abs), cmap="RdBu_r", show=False)[0]
            fig_topo.colorbar(im_eo, ax=axes_topo[0], orientation="vertical", shrink=0.7, label="% Change")
            
            # 2. EC Change Topomap
            axes_topo[1].set_title("EC Alpha Power % Change")
            im_ec = mne.viz.plot_topomap(ec_pct_plot, info_topo, axes=axes_topo[1], vlim=(-v_power_abs, v_power_abs), cmap="RdBu_r", show=False)[0]
            fig_topo.colorbar(im_ec, ax=axes_topo[1], orientation="vertical", shrink=0.7, label="% Change")
            
            # 3. Reactivity Change Topomap
            axes_topo[2].set_title("Reactivity % Change")
            v_react_abs = max(abs(np.nanmin(react_pct_plot)), abs(np.nanmax(react_pct_plot))) if len(react_pct_plot) > 0 else 10.0
            im_react = mne.viz.plot_topomap(react_pct_plot, info_topo, axes=axes_topo[2], vlim=(-v_react_abs, v_react_abs), cmap="RdBu_r", show=False)[0]
            fig_topo.colorbar(im_react, ax=axes_topo[2], orientation="vertical", shrink=0.7, label="% Change")
            
            plt.suptitle(f"EEG Alpha Metrics Percentage Change Topomaps - Day {day}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            
            topo_path = output_dir / f"day{day}_pre_post_comparison_topomap.svg"
            plt.savefig(topo_path, dpi=300)
            plt.close()
            logger.info(f"Saved comparison topoplots to {topo_path}")
            
    except Exception as e:
        logger.error(f"Failed to analyze Day {day}: {e}", exc_info=True)

# %%
