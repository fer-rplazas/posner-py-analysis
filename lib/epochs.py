import numpy as np


class Epochs:
    def __init__(
        self,
        eeg_data,
        eeg_names: list[str],
        trigger_data,
        trigger_ch_names: list[str],
        other_data,
        other_ch_names: list[str],
        fs: float,
        event_timestamps,
        events,
        epoch_lims: tuple[float, float] = (-3, 5),
    ):

        # Data has shape (n_epochs, n_channels, n_samples)

        assert eeg_data.shape[1] == len(eeg_names)
        assert trigger_data.shape[1] == len(trigger_ch_names)
        assert other_data.shape[1] == len(other_ch_names)

        assert (
            eeg_data.shape[0]
            == trigger_data.shape[0]
            == other_data.shape[0]
            == event_timestamps.shape[0]
        ), "All data must have the same number of epochs as event timestamps"

        assert (
            eeg_data.shape[2] == trigger_data.shape[2] == other_data.shape[2]
        ), "All data must have the same number of samples"

        self.n_epochs = eeg_data.shape[0]

        self.eeg_data = eeg_data
        self.eeg_names = eeg_names
        self.trigger_data = trigger_data
        self.trigger_ch_names = trigger_ch_names
        self.other_data = other_data
        self.other_ch_names = other_ch_names

        self.fs = fs
        self.epoch_lims = epoch_lims
        self.t = (
            np.linspace(0, (self.eeg_data.shape[-1] - 1) / fs, self.eeg_data.shape[-1])
            + epoch_lims[0]
        )

        assert (
            event_timestamps.shape[1] == 4
        ), "event_timestamps must contain indices for 4 events per epoch"
        assert (
            events.shape[0] == event_timestamps.shape[0]
        ), "events must have the same number of entries as event_timestamps"
        assert events.shape[1] == 4, "events must contain 4 event codes per epoch"

        self.event_timestamps = event_timestamps
        self.events = events

        # For each epoch, compute RTs from t and event timestamps 3 and 4
        rts = []
        for event_timestamp in event_timestamps:
            rts.append(self.t[event_timestamp[3]] - self.t[event_timestamp[2]])
        self.rts = np.array(rts)

    def select_epochs_from_indices(self, indices: list[int]):
        return Epochs(
            self.eeg_data[indices],
            self.eeg_names,
            self.trigger_data[indices],
            self.trigger_ch_names,
            self.other_data[indices],
            self.other_ch_names,
            self.fs,
            self.event_timestamps[indices],
            self.events[indices],
            self.epoch_lims,
        )

    def indices_from_query(self, seqs: list[list[str]]):
        selected_indices = []
        for ix, event_seq in enumerate(self.events):
            for seq in seqs:
                if event_seq.tolist() == seq:
                    selected_indices.append(ix)
                    break
        return selected_indices

    def query(self, query_str: str):
        if query_str == "mistakes":
            seqs = [
                ["fixation_cross", "left", "left", "right"],
                ["fixation_cross", "left", "right", "left"],
                ["fixation_cross", "right", "right", "left"],
                ["fixation_cross", "right", "left", "right"],
                ["fixation_cross", "both", "left", "right"],
                ["fixation_cross", "both", "right", "left"],
            ]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))

        elif query_str == "congruent-left":  # Arrow points left, target appears on the left
            seqs = [["fixation_cross", "left", "left", "left"]]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "congruent-right":
            seqs = [["fixation_cross", "right", "right", "right"]]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "congruent-all":
            seqs = [
                ["fixation_cross", "right", "right", "right"],
                ["fixation_cross", "left", "left", "left"],
            ]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "incongruent-left":  # Arrow points right, but target appears on the left
            seqs = [["fixation_cross", "right", "left", "left"]]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "incongruent-right":  # Arrow points left, but target appears on the right
            seqs = [["fixation_cross", "left", "right", "right"]]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "incongruent-all":
            seqs = [
                ["fixation_cross", "left", "right", "right"],
                ["fixation_cross", "right", "left", "left"],
            ]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "neutral-left":  # Arrow points both ways, target on the left
            seqs = [["fixation_cross", "both", "left", "left"]]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "neutral-right":  # Arrow points both ways, target on the right
            seqs = [["fixation_cross", "both", "right", "right"]]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))
        elif query_str == "neutral-all":
            seqs = [
                ["fixation_cross", "both", "left", "left"],
                ["fixation_cross", "both", "right", "right"],
            ]
            return self.select_epochs_from_indices(self.indices_from_query(seqs))

        return None
