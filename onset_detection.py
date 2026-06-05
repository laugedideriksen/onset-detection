import os
import csv
import librosa
import numpy as np
from scipy.signal import medfilt
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter, median_filter
import matplotlib.pyplot as plt


class OnsetDetect:
    def __init__(
        self, sound_file: str, title="", start: float = 0, end: float | None = None
    ) -> None:
        """Load soundfile and create OnsetDetect object"""
        self.sound_file = sound_file
        self.start = start
        self.end = end
        duration = (self.end - self.start) if self.end else None
        self._array, self._sample_rate = librosa.load(
            sound_file, offset=self.start, duration=duration
        )
        self.title = title

    @property
    def sound_file(self):
        return self._sound_file

    @sound_file.setter
    def sound_file(self, value):
        if os.path.isfile(value):
            self._sound_file = value
        else:
            raise FileNotFoundError(f"{self.sound_file} doesn't exist or isn't a file.")

    @property
    def start(self):
        """The start property."""
        return self._start

    @start.setter
    def start(self, value):
        if value < 0:
            raise ValueError("The value of start must be 0 or greater.")
        self._start = value

    @property
    def end(self):
        """The end property."""
        return self._end

    @end.setter
    def end(self, value):
        if value and value < 0:
            raise ValueError("The value of end must be greater than 0.")
        elif value and value < self.start:
            raise ValueError("The value of end must be greater than the value of start")
        self._end = value

    # ENVELOPES
    # The output of all envelopes is normalised.

    def _envelope_spectral_flux(self):
        """Compute an envelope based on spectral flux"""
        spectral_flux_envelope = librosa.onset.onset_strength(
            y=self._array, sr=self._sample_rate
        )
        spectral_flux_envelope = spectral_flux_envelope / (
            np.max(spectral_flux_envelope) + 1e-6
        )
        return spectral_flux_envelope

    def _envelope_diff_spectral_flux(self):
        """Compute an envelope that is the derivative of the spectral flux envelope"""
        spectral_flux_envelope = librosa.onset.onset_strength(
            y=self._array, sr=self._sample_rate
        )
        diff_specflux_envelope = np.diff(spectral_flux_envelope)
        diff_specflux_envelope = np.maximum(0, diff_specflux_envelope)
        diff_specflux_envelope = np.concatenate([[0], diff_specflux_envelope])
        diff_specflux_envelope = diff_specflux_envelope / (
            np.max(diff_specflux_envelope) + 1e-6
        )
        return diff_specflux_envelope

    def _envelope_diff_rms(self, frame_length=2048):
        """Compute an envelope that is the derivative of a root-mean-square envelope"""
        rms = librosa.feature.rms(
            y=self._array, frame_length=frame_length, hop_length=512
        )[0]
        rms_diff = np.diff(rms)
        rms_diff_rising = np.maximum(0, rms_diff)
        rms_diff_envelope = np.concatenate([[0], rms_diff_rising])
        rms_diff_envelope = rms_diff_envelope / (np.max(rms_diff_envelope) + 1e-6)
        return rms_diff_envelope

    def _envelope_delta_rms(self, frame_length=2048):
        """Compute an envelope based on the delta between frames"""
        rms = librosa.feature.rms(
            y=self._array, frame_length=frame_length, hop_length=512
        )[0]
        delta_window = 4
        smoothed_rms = np.convolve(
            rms, np.ones(delta_window) / delta_window, mode="same"
        )
        rms_delta_envelope = rms - smoothed_rms
        rms_delta_envelope = np.maximum(0, rms_delta_envelope)
        rms_delta_envelope = rms_delta_envelope / (np.max(rms_delta_envelope) + 1e-6)
        return rms_delta_envelope

    def _envelope_chroma_cqt(self):
        """Compute envelope based on a constant-Q chromagram"""
        chroma_cqt = librosa.feature.chroma_cqt(
            y=self._array, sr=self._sample_rate, hop_length=512
        )
        chroma_cqt_diff = np.diff(chroma_cqt)
        chroma_cqt_diff = np.maximum(0, chroma_cqt_diff).mean(axis=0)
        chroma_envelope = np.concatenate([[0], chroma_cqt_diff])
        chroma_envelope = chroma_envelope / (np.max(chroma_envelope) + 1e-6)
        return chroma_envelope

    def _envelope_hybrid(self, *envelopes):
        """Create a hybrid envelope by adding together normalised envelopes"""
        hybrid_envelope = []
        for i in envelopes:
            if i == "spectral_flux":
                envelope = np.array(self._envelope_spectral_flux())
            elif i == "diff_spectral_flux":
                envelope = np.array(self._envelope_diff_spectral_flux())
            elif i == "diff_rms":
                envelope = np.array(self._envelope_diff_rms())
            elif i == "delta_rms":
                envelope = np.array(self._envelope_delta_rms())
            elif i == "chroma_cqt":
                envelope = np.array(self._envelope_chroma_cqt())
            hybrid_envelope.append(envelope)
        hybrid_envelope = np.sum(np.vstack(hybrid_envelope), axis=0)
        hybrid_envelope = hybrid_envelope / (np.max(hybrid_envelope) + 1e-6)
        return hybrid_envelope

    # FILTERING
    def _median_filter(self, onset_envelope, kernel_size=3):
        onset_envelope = medfilt(onset_envelope, kernel_size=kernel_size)
        return onset_envelope

    # THRESHOLD

    def _global_mad_threshold(self, onset_envelope, k_factor=2.0):
        """Compute a global threshold"""
        median = np.median(onset_envelope)
        mad = median_abs_deviation(onset_envelope)
        k = k_factor
        threshold = median + (k * mad)
        threshold = max(threshold, 0)
        return threshold

    def _moving_mad_threshold(self, onset_envelope, window_duration=2.0, k_factor=2.0):
        """Compute moving threshold."""
        window = int(window_duration * self._sample_rate / 512)
        window_median = median_filter(onset_envelope, size=window, mode="nearest")
        window_mad = generic_filter(
            onset_envelope, median_abs_deviation, size=window, mode="nearest"
        )
        threshold = window_median + (window_mad * k_factor)
        return threshold

    # ONSET DETECTION

    def _centroid_peak_pick(self, onset_envelope, threshold):
        """Pick loudest frame of detected event"""
        indices_above_threshold = np.where(onset_envelope > threshold)[0]
        detected_onset_frames = []
        if len(indices_above_threshold) > 0:
            current_event = [indices_above_threshold[0]]
            for i in range(1, len(indices_above_threshold)):
                idx = indices_above_threshold[i]
                previous_idx = indices_above_threshold[i - 1]
                if idx - previous_idx <= 1:
                    current_event.append(idx)
                else:
                    loudest_frame = max(current_event, key=lambda f: onset_envelope[f])
                    detected_onset_frames.append(loudest_frame)
                    current_event = [idx]
        loudest_frame = max(current_event, key=lambda f: onset_envelope[f])
        detected_onset_frames.append(loudest_frame)
        detected_onset_times = librosa.frames_to_time(
            np.array(detected_onset_frames), sr=self._sample_rate
        )
        return detected_onset_times

    def _backtrack_peak_pick(self, onset_envelope, threshold):
        """Peak pick using threshold, but backtrack to local minimum on entire envelope"""
        indices_above_threshold = np.where(onset_envelope > threshold)[0]
        detected_onset_frames = []
        if len(indices_above_threshold) > 0:
            current_event = [indices_above_threshold[0]]
            for i in range(1, len(indices_above_threshold)):
                idx = indices_above_threshold[i]
                previous_idx = indices_above_threshold[i - 1]
                if idx - previous_idx <= 1:
                    current_event.append(idx)
                else:
                    j = current_event[0]
                    while (onset_envelope[j] - onset_envelope[j - 1]) > 0:
                        j -= 1
                    detected_onset_frames.append(j)
                    current_event = [idx]
        j = current_event[0]
        while (onset_envelope[j] - onset_envelope[j - 1]) > 0:
            j -= 1
        detected_onset_frames.append(j)
        detected_onset_times = librosa.frames_to_time(
            np.array(detected_onset_frames), sr=self._sample_rate
        )
        return detected_onset_times

    def _librosa_peak_pick(self, onset_envelope):
        """Use Librosa's built-in peak picker. Note: this disables the threshold filter"""
        onset_timestamps = librosa.util.peak_pick(
            onset_envelope,
            pre_max=5,
            post_max=5,
            pre_avg=2,
            post_avg=2,
            delta=0.7,
            wait=2,
        )
        onset_timestamps = librosa.frames_to_time(onset_timestamps)
        threshold = 0
        return onset_timestamps, threshold

    def _merge_onsets(self, onset_timestamps, min_note_gap):
        """Merge near-simultaneous detections. Usually best left disabled."""
        minimum_note_gap = min_note_gap  # in miliseconds

        final_onset_timestamps = []
        try:
            detected_onset_times = np.sort(onset_timestamps)
            current_onset_cluster = [detected_onset_times[0]]

            for i in range(1, len(detected_onset_times)):
                timestamp = detected_onset_times[i]
                previous_timestamp = detected_onset_times[i - 1]

                if timestamp - previous_timestamp < minimum_note_gap:
                    current_onset_cluster.append(timestamp)
                else:
                    centroid_onset = np.mean(current_onset_cluster)
                    final_onset_timestamps.append(centroid_onset)
                    current_onset_cluster = [timestamp]
            final_onset_timestamps.append(np.mean(current_onset_cluster))
            return final_onset_timestamps
        except Exception as e:
            raise e

    # OUTPUT

    def _output(self, output_type, output_destination, onset_timestamps):
        """Output onset timestamps as Python list, as CSV-liske rows, or as a CSV file."""
        onset_timestamps = np.array(onset_timestamps) + self.start
        onset_timestamps = onset_timestamps.tolist()
        if output_type == "list":
            print(onset_timestamps)
        elif output_type == "rows":
            for idx, ts in enumerate(onset_timestamps):
                print(f"{ts},{idx}")
        elif output_type == "csv":
            if output_destination:
                output_destination = f"{output_destination}.csv"
            else:
                output_destination = "onset_timestamps.csv"
            with open(output_destination, "w", newline="") as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=" ")
                for idx, ts in enumerate(onset_timestamps):
                    csvwriter.writerow([ts, idx])

    # VISUALISATION

    def _plot(
        self, onset_timestamps, onset_envelope, threshold, threshold_type, peak_picking
    ):
        """Plot spectrogram, envelope, threshold (if applicable), and onsets"""
        times = librosa.frames_to_time(
            np.arange(len(onset_envelope)), sr=self._sample_rate
        )

        D = np.abs(librosa.stft(self._array))
        _, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            x_axis="time",
            y_axis="log",
            ax=ax[0],
            sr=self._sample_rate,
        )

        _, ymax = ax[0].get_ylim()

        # Generate spectrogram with onset lines
        ax[0].set(title=f"{self.title}")
        ax[0].label_outer()
        ax[0].vlines(
            onset_timestamps,
            0,
            ymax,
            color="g",
            alpha=0.9,
            linewidth=1,
            linestyles="solid",
            label="Onsets",
        )

        # Graph onset envelope
        ax[1].plot(times, onset_envelope, label="Onset envelope")
        # ax[1].plot(times, np.insert(np.diff(onset_envelope), 0, 0, axis=0), label='env_diff')

        ax[1].vlines(
            onset_timestamps,
            0,
            np.max(onset_envelope),
            color="g",
            linewidth=2,
            label="Detected onsets",
        )

        if peak_picking != "librosa":
            if threshold_type == "global":
                ax[1].axhline(y=threshold, color="r", label="Threshold")
            elif threshold_type == "moving":
                ax[1].plot(times, threshold, color="r", label="Threshold")

        plt.legend()
        plt.show()

    def _plot_envelope_comparison(
        self, times, *envelopes, global_threshold, moving_threshold
    ):
        """Plot all non-hybrid envelopes for comparison."""
        D = np.abs(librosa.stft(self._array))
        _, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            x_axis="time",
            y_axis="log",
            ax=ax[0],
            sr=self._sample_rate,
        )

        spacing = 1
        for envelope in envelopes:
            ax[1].plot(times, envelope[0], color=envelope[2])
            ax[1].plot(
                times, spacing + envelope[0], label=envelope[1], color=envelope[2]
            )
            spacing += 1
        ax[1].axhline(
            y=global_threshold, color="r", label="Global threshold", linewidth=1
        )
        ax[1].plot(
            times, moving_threshold, color="r", label="Moving threshold", linewidth=1
        )
        plt.legend()
        plt.show()

    def _plot_filter_comparison(self, times, *envelopes, filter_kernel):
        """Plot filtered and unfiltered envelopes."""
        D = np.abs(librosa.stft(self._array))
        _, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            x_axis="time",
            y_axis="log",
            ax=ax[0],
            sr=self._sample_rate,
        )

        for envelope in envelopes:
            ax[1].plot(times, 1 + envelope[0], label=envelope[1], color=envelope[2])

        for envelope in envelopes:
            ax[1].plot(
                times, self._median_filter(envelope[0], kernel_size=filter_kernel)
            )

        plt.legend()
        plt.show()
        # TODO: mod 1 or remove y-axis labels in lower table.

    # PUBLIC METHODS

    def detect_onsets(
        self,
        envelope: str = "spectral_flux",
        hybrid_env_components: list = ["spectral_flux", "delta_rms"],
        filtering: str = "median_filter",
        filter_kernel: int = 3,
        threshold_k: float = 0.9,
        threshold_type: str = "global",
        threshold_window: float = 1.0,
        peak_picking: str = "backtrack",
        merge_onsets: bool = False,
        min_note_gap: float = 0.08,
        output: str = "list",
        output_destination: str | bool = False,
        plot: bool = True,
    ) -> str | None:
        """Select envelope, adjust parameters, detect onsets, and plot results."""
        if envelope == "spectral_flux":
            onset_envelope = self._envelope_spectral_flux()
        elif envelope == "diff_spectral_flux":
            onset_envelope = self._envelope_diff_spectral_flux()
        elif envelope == "diff_rms":
            onset_envelope = self._envelope_diff_rms()
        elif envelope == "delta_rms":
            onset_envelope = self._envelope_delta_rms()
        elif envelope == "chroma_cqt":
            onset_envelope = self._envelope_chroma_cqt()
        elif envelope == "hybrid":
            onset_envelope = self._envelope_hybrid(*hybrid_env_components)
        else:
            raise ValueError(f"{envelope} is not a supported envelope type.")

        if filtering == "none":
            pass
        elif filtering == "median_filter":
            onset_envelope = self._median_filter(
                onset_envelope=onset_envelope, kernel_size=filter_kernel
            )

        if threshold_type == "global":
            threshold = self._global_mad_threshold(
                onset_envelope=onset_envelope, k_factor=threshold_k
            )
        elif threshold_type == "moving":
            threshold = self._moving_mad_threshold(
                onset_envelope=onset_envelope,
                k_factor=threshold_k,
                window_duration=threshold_window,
            )
        else:
            raise ValueError(f"{threshold_type} is not a supported threshold type.")

        if peak_picking == "centroid":
            onset_timestamps = self._centroid_peak_pick(
                onset_envelope=onset_envelope, threshold=threshold
            )
        elif peak_picking == "backtrack":
            onset_timestamps = self._backtrack_peak_pick(
                onset_envelope=onset_envelope, threshold=threshold
            )
        elif peak_picking == "librosa":
            onset_timestamps, threshold = self._librosa_peak_pick(
                onset_envelope=onset_envelope
            )

        if merge_onsets:
            onset_timestamps = self._merge_onsets(
                onset_timestamps=onset_timestamps, min_note_gap=min_note_gap
            )

        self._output(
            output_type=output,
            output_destination=output_destination,
            onset_timestamps=onset_timestamps,
        )

        if plot:
            self._plot(
                onset_timestamps=onset_timestamps,
                onset_envelope=onset_envelope,
                threshold=threshold,
                threshold_type=threshold_type,
                peak_picking=peak_picking,
            )

    def compare(
        self,
        compare_parameter: str = "envelopes",
        threshold_k: float = 2.0,
        threshold_window: float = 2.0,
        filter_kernel: int = 3,
    ) -> None:
        """Compare envelopes."""
        specflux = self._envelope_spectral_flux()
        specflux_diff = self._envelope_diff_spectral_flux()
        rms_diff = self._envelope_diff_rms()
        rms_delta = self._envelope_delta_rms()
        chroma_cqt = self._envelope_chroma_cqt()
        times = librosa.frames_to_time(np.arange(len(specflux)), sr=self._sample_rate)

        envelopes = [
            (specflux, "Spectral flux", "C0"),
            (specflux_diff, "Spectral flux (derivative)", "C1"),
            (rms_diff, "Root-mean-square (derivative)", "C2"),
            (rms_delta, "Root-mean-square (delta)", "C3"),
            (chroma_cqt, "CQT chroma", "C4"),
        ]

        sum_onset = self._envelope_hybrid(
            *[
                "spectral_flux",
                "diff_spectral_flux",
                "diff_rms",
                "delta_rms",
                "chroma_cqt",
            ]
        )

        global_threshold = self._global_mad_threshold(
            onset_envelope=sum_onset, k_factor=threshold_k
        )
        moving_threshold = self._moving_mad_threshold(
            onset_envelope=sum_onset,
            k_factor=threshold_k,
            window_duration=threshold_window,
        )

        if compare_parameter == "envelopes":
            self._plot_envelope_comparison(
                times,
                *envelopes,
                global_threshold=global_threshold,
                moving_threshold=moving_threshold,
            )
        elif compare_parameter == "filtering":
            self._plot_filter_comparison(times, *envelopes, filter_kernel=filter_kernel)
        else:
            raise ValueError(f"{compare_parameter} is not a valid parameter to compare")
