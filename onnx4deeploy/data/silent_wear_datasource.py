"""
SilentWear DataSource — loads real EMG windows from the SilentWear dataset.

Dataset: https://huggingface.co/datasets/PulpBio/SilentWear
Paper:   Spacone et al., "SilentWear: an Ultra-Low Power Wearable System for
         EMG-based Silent Speech Recognition", arXiv: 2603.02847.

Data format (data_raw_and_filt):
  - One .h5 file per batch, key="emg", stored as a pandas DataFrame
  - Each row = one time sample at 500 Hz
  - 14 filtered EMG channels: Ch_0_filt ... Ch_10_filt, Ch_13_filt, Ch_14_filt, Ch_15_filt
    (Ch_11 and Ch_12 are unused in the hardware design)
  - Label_int column: integer class label per sample (0=rest, 1-8=commands)
  - Labels are constant within a contiguous segment (one word repetition)

Windowing:
  - The continuous recording is segmented at label change points
  - One fixed-length window of `window_samples` samples is extracted from
    the center of each segment
  - Window shape after reshaping: (1, 1, 14, window_samples) = SpeechNet input
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_datasource import DataSource

# The 14 differential EMG channels used by SpeechNet.
# Ch_11 and Ch_12 are absent from the dataset — the hardware skips them.
EMG_FILT_COLS = [
    "Ch_0_filt",  "Ch_1_filt",  "Ch_2_filt",  "Ch_5_filt",
    "Ch_3_filt",  "Ch_4_filt",  "Ch_7_filt",  "Ch_6_filt",
    "Ch_8_filt",  "Ch_15_filt", "Ch_9_filt",  "Ch_14_filt",
    "Ch_10_filt", "Ch_13_filt",
]  # 14 channels, hardware order [0,1,2,5,3,4,7,6,8,15,9,14,10,13]


class SilentWearDataSource(DataSource):
    """
    Loads real EMG windows from the SilentWear dataset for training
    mini-batch generation.

    Reads from the `data_raw_and_filt` folder of the dataset, which contains
    continuous filtered EMG recordings. Segmentation and windowing are
    performed on-the-fly inside load_batches().

    Args:
        data_path:       Path to the `data_raw_and_filt` folder.
        subject:         Subject ID, e.g. "S01", "S02", "S03", "S04".
        session:         Recording session number (1, 2, or 3).
                         Session 3 = most recent day, best for fine-tuning
                         since it simulates a new deployment scenario.
        batch:           Batch number within the session (1 to 5).
        condition:       "vocalized" or "silent".
        window_samples:  Number of time samples per window.
                         700 = 1400ms @ 500Hz (default, matches SpeechNet).
                         400 = 800ms @ 500Hz (deployed inference size).
    """

    def __init__(
        self,
        data_path: str,
        subject: str = "S01",
        session: int = 1,
        batch: int = 1,
        condition: str = "vocalized",
        window_samples: int = 700,
        downsample_rest: bool = False,
    ):
        self.data_path = Path(data_path)
        self.subject = subject
        self.session = session
        self.batch = batch
        self.condition = condition
        self.window_samples = window_samples
        self.downsample_rest = downsample_rest

        # Cache: populated on first call to _load_windows()
        # so the h5 file is only read once even if load_batches() is called
        # multiple times.
        self._inputs: Optional[List[np.ndarray]] = None
        self._labels: Optional[List[np.ndarray]] = None

    def _load_windows(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Read one batch h5 file, segment it by label, and extract one
        fixed-length window per segment.

        Returns:
            inputs: list of float32 arrays, each shape (1, 1, 14, window_samples)
            labels: list of int64 arrays, each shape (1,)
        """
        # Return cached result if already loaded
        if self._inputs is not None:
            return self._inputs, self._labels

        # Build path:
        # data_raw_and_filt/S01/vocalized/sess_1_batch_1.h5
        h5_path = (
            self.data_path
            / self.subject
            / self.condition
            / f"sess_{self.session}_batch_{self.batch}.h5"
        )

        print(f"   Loading SilentWear data from {h5_path} ...")
        df = pd.read_hdf(h5_path, key="emg")

        # Extract the 14 filtered channels as a (N_samples, 14) float32 array.
        # We use the filtered signals (Ch_i_filt) because SpeechNet was trained
        # on pre-filtered data (20Hz high-pass + 50Hz notch already applied).
        emg = df[EMG_FILT_COLS].values.astype(np.float32)  # (N_samples, 14)

        # Extract the integer label for each time sample.
        # Within one word repetition, all samples share the same label.
        label_col = df["Label_int"].values  # (N_samples,)

        inputs: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        # Find indices where the label changes — these are segment boundaries.
        # np.diff gives the difference between consecutive elements;
        # non-zero entries indicate a label transition.
        # +1 shifts the index to the first sample of the new segment.
        change_idx = np.where(np.diff(label_col) != 0)[0] + 1

        # seg_starts[i] and seg_ends[i] define the i-th segment's time range.
        seg_starts = np.concatenate([[0], change_idx])
        seg_ends   = np.concatenate([change_idx, [len(label_col)]])

        for start, end in zip(seg_starts, seg_ends):
            seg = emg[start:end]       # (seg_len, 14) — one word/rest segment
            lbl = label_col[start]     # scalar label for this segment

            # Skip segments shorter than the required window.
            # This can happen for very short rest periods at recording edges.
            if len(seg) < self.window_samples:
                continue

            # Extract a window from the CENTER of the segment.
            # The center is preferred over the onset because:
            # - Onset has reaction-time jitter (paper limitation §V)
            # - Offset has trailing muscle relaxation
            # - Center captures the steady-state articulation
            offset = (len(seg) - self.window_samples) // 2
            window = seg[offset : offset + self.window_samples]  # (window_samples, 14)

            # Per-window scalar normalization: subtract mean and divide by std
            # computed across all 14×window_samples values (matches fine-tuning preprocessing).
            mean = window.mean()
            std  = window.std()
            window = (window - mean) / (std + 1e-8)

            # Reshape to SpeechNet input format: (batch, in_channels, height, width)
            # = (1, 1, 14, window_samples)
            # .T transposes (window_samples, 14) → (14, window_samples)
            # [np.newaxis, np.newaxis] adds the batch and channel dimensions
            window = window.T[np.newaxis, np.newaxis, :, :]  # (1, 1, 14, window_samples)

            inputs.append(window)
            labels.append(np.array([int(lbl)], dtype=np.int64))  # shape (1,)

        print(f"   Extracted {len(inputs)} windows "
              f"(subject={self.subject}, session={self.session}, "
              f"batch={self.batch}, condition={self.condition}, "
              f"window={self.window_samples} samples)")

        if self.downsample_rest and len(inputs) > 0:
            from collections import Counter
            label_ints = [int(l[0]) for l in labels]
            counts = Counter(label_ints)
            cmd_counts = [v for k, v in counts.items() if k != 0]
            if cmd_counts:
                min_cmd = min(cmd_counts)
                rest_idx = [i for i, lbl in enumerate(label_ints) if lbl == 0]
                if len(rest_idx) > min_cmd:
                    keep_rest = np.random.RandomState(42).choice(
                        rest_idx, size=min_cmd, replace=False
                    ).tolist()
                    non_rest_idx = [i for i, lbl in enumerate(label_ints) if lbl != 0]
                    keep = sorted(non_rest_idx + keep_rest)
                    inputs = [inputs[i] for i in keep]
                    labels = [labels[i] for i in keep]
                    print(f"   Rest downsampled: {len(rest_idx)} → {min_cmd} "
                          f"(matches min command class count)")

        # Cache for reuse
        self._inputs = inputs
        self._labels = labels

        return inputs, labels

    def load_batches(
        self,
        n_batches: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return n_batches (input, label) pairs sampled from the h5 file.

        Follows the same interface as MNISTDataSource.load_batches():
            inputs_list: n_batches float32 arrays, each shape == input_shape
            labels_list: n_batches int64 arrays,   each shape == (batch_size,)

        If n_batches > number of available windows, sampling is done with
        replacement (the same window may appear multiple times). This matches
        the epoch-cycling behaviour of MNISTDataSource.

        Args:
            n_batches:   Number of mini-batches to return.
            input_shape: Expected shape of each input tensor, e.g. (1,1,14,700).
                         Used only for the assertion check below.
            num_classes: Number of output classes (9 for SpeechNet).
                         Not used directly — labels come from the data.
            seed:        Random seed for reproducible sampling.

        Returns:
            inputs_list: list of n_batches float32 arrays
            labels_list: list of n_batches int64 arrays
        """
        inputs, labels = self._load_windows()

        assert len(inputs) > 0, (
            f"No windows extracted. Check path: "
            f"{self.data_path / self.subject / self.condition}"
        )

        # Verify the window shape matches what the exporter expects.
        # Catches mismatches early (e.g. window_samples=700 but exporter uses 400).
        assert inputs[0].shape == input_shape, (
            f"Shape mismatch: data source produces {inputs[0].shape} "
            f"but exporter expects {input_shape}. "
            f"Check window_samples vs time_steps in config."
        )

        # Use a local RandomState — same pattern as MNISTDataSource —
        # so we never modify the global numpy random state.
        rng = np.random.RandomState(seed)

        # Sample n_batches indices from the available windows.
        # replace=True when n_batches > len(inputs) to avoid errors.
        idx = rng.choice(
            len(inputs),
            size=n_batches,
            replace=len(inputs) < n_batches,
        )

        return [inputs[i] for i in idx], [labels[i] for i in idx]