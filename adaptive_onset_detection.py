import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter, median_filter

class OnsetDetect:
    """Takes a sound file and optionally a score file. Has methods to detect onsets and can output as list, or as csv with pitch name and note value."""

    def __init__(
        self, sound_file: str, score_file: str = None, title: str = None
    ) -> None:
        self._sound_file = sound_file
        self._score_file = score_file
        self._array, self._sample_rate = librosa.load(sound_file, duration=15)

    @property
    def sound_file(self):
        """The sound_file property."""
        return self._sound_file

    @sound_file.setter
    def sound_file(self, value):
        if os.path.isfile(self.sound_file):
            self._sound_file = value
        else:
            raise FileNotFoundError(f"{self.sound_file} doesn't exist or isn't a file.")

    @property
    def score_file(self):
        """The score_file property."""
        return self._score_file

    @score_file.setter
    def score_file(self, value):
        if self.score_file is None:
            pass
        elif os.path.isfile(self.score_file):
            self._score_file = value
        else:
            raise FileNotFoundError(f"{self.score_file} doesn't exist or isn't a file.")

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        self._array = value

    @property
    def sample_rate(self):
        return self._sample_rate

    def _preprocess(self, threshold, onset_envelope, diff_envelope):
        onset_frames = librosa.util.peak_pick(
                onset_envelope,
                pre_avg=3,
                post_avg=3,
                pre_max=2,
                post_max=2,
                wait=5,
                delta=0.25,
                )
        print(onset_envelope, onset_frames)

        median_floor = np.median(onset_envelope)
        calculated_threshold = threshold * median_floor

        picked_peaks = []
        for frame in onset_frames:
            if onset_envelope[frame] > calculated_threshold:
                picked_peaks.append(frame)
        picked_peaks = np.array(picked_peaks)

        return picked_peaks, onset_frames, calculated_threshold


    def _plot(self, times_in_envelope, times, onset_envelope, raw_onset_frames, onset_frames, threshold, calculated_threshold):
        D = np.abs(librosa.stft(self.array))
        fig, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            x_axis="time",
            y_axis="log",
            ax=ax[0],
            sr=self.sample_rate,
        )

        ax[0].set(title=f"Power spectrogram: {self.sound_file}")
        ax[0].label_outer()
        ax[0].vlines(
            times,
            0,
            16384,
            color="g",
            alpha=0.9,
            linewidth=1,
            linestyles="solid",
            label="Onsets",
        )

        ax[1].plot(times_in_envelope, onset_envelope, label='Raw Onset Envelope', alpha=0.6)
        ax[1].vlines(times, 0, np.max(onset_envelope), colors='green', linestyles='solid', linewidth=1, label='Valid Onsets')
        if threshold:
            ax[1].axhline(y=calculated_threshold, color='r', linestyle='--', label='Threshold')
            rejected_frames = [f for f in raw_onset_frames if f not in onset_frames]
            if len(rejected_frames) > 0:
                ax[1].vlines(times_in_envelope[rejected_frames], 0, np.max(onset_envelope), colors='orange', linestyles='dotted', linewidth=1, label='Rejected (Below Threshold)')
            ax[1].legend()
        plt.show()

    # ENVELOPES

    def _envelope_spectral_flux(self, frame_length=2048):
        # Compute two envelopes: spectral flux and root-mean-square derivative.
        spectral_flux_envelope = librosa.onset.onset_strength(y=self.array, sr=self.sample_rate)
        spectral_flux_envelope = spectral_flux_envelope / (np.max(spectral_flux_envelope) + 1e-6)
        return spectral_flux_envelope

    def _envelope_diff_spectral_flux(self):
        spectral_flux_envelope = librosa.onset.onset_strength(y=self.array, sr=self.sample_rate)
        diff_specflux_envelope = np.diff(spectral_flux_envelope)
        diff_specflux_envelope = np.maximum(0, diff_specflux_envelope)
        diff_specflux_envelope = np.concatenate([[0], diff_specflux_envelope])
        diff_specflux_envelope = diff_specflux_envelope / (np.max(diff_specflux_envelope) + 1e-6)
        return diff_specflux_envelope

    def _envelope_differential_rms(self, frame_length=2048):
        rms = librosa.feature.rms(y=self.array, frame_length=frame_length, hop_length=512)[0]
        rms_diff = np.diff(rms)
        rms_diff_rising = np.maximum(0, rms_diff)
        rms_diff_envelope = np.concatenate([[0], rms_diff_rising])
        rms_diff_envelope = rms_diff_envelope / (np.max(rms_diff_envelope) + 1e-6)
        return rms_diff_envelope

    def _envelope_delta_rms(self, frame_length=2048):
        rms = librosa.feature.rms(y=self.array, frame_length=frame_length, hop_length=512)[0]
        delta_window = 4
        smoothed_rms = np.convolve(rms, np.ones(delta_window)/delta_window, mode='same')
        rms_delta_envelope = rms - smoothed_rms
        rms_delta_envelope = np.maximum(0, rms_delta_envelope)
        rms_delta_envelope = rms_delta_envelope / (np.max(rms_delta_envelope) + 1e-6)
        return rms_delta_envelope

    def _envelope_chroma_cqt(self):
        chroma_cqt = librosa.feature.chroma_cqt(y=self.array, sr=self.sample_rate, hop_length=512)
        chroma_cqt_diff = np.diff(chroma_cqt)
        chroma_cqt_diff = np.maximum(0, chroma_cqt_diff).mean(axis=0)
        chroma_envelope = np.concatenate([[0], chroma_cqt_diff])
        chroma_envelope = chroma_envelope / (np.max(chroma_envelope) + 1e-6)
        return chroma_envelope * 0.5 # Multiplied by 0.5 to weight it lower in hybrid envelopes, as it is quite jittery. If used on its own, the multiplication is in consequential.

    def _create_hybrid_envelope(self, *envelopes):
        hybrid_envelope = [0]
        for i in envelopes:
            hybrid_envelope += i
        return hybrid_envelope

    # THRESHOLD

    def _global_mad_threshold(self, onset_envelope, k_factor=2.0):
        # Compute an adaptive threshold
        median = np.median(onset_envelope)
        mad = median_abs_deviation(onset_envelope)
        k = k_factor
        threshold = median + (k * mad)
        threshold = max(threshold, 0)
        return threshold

    def _moving_mad_threshold(self, onset_envelope, window_duration=2.0, k_factor=2.0):
        window = int(window_duration * self.sample_rate/512)
        window_median = median_filter(onset_envelope, size=window, mode='nearest')
        window_mad = generic_filter(onset_envelope, median_abs_deviation, size=window, mode='nearest')
        threshold = window_median + (window_mad * k_factor)
        return threshold

    def _pick_and_merge(self, onset_envelope, threshold):
        onset_envelope = medfilt(onset_envelope, kernel_size=7)
        indices_above_threshold = np.where(onset_envelope > threshold)[0]

        # Peak Picking
        detected_onset_frames = []
        try:
            current_event = indices_above_threshold[0]
            for i in range(1, len(indices_above_threshold)):
                idx = indices_above_threshold[i]
                previous_idx = indices_above_threshold[i-1]
                if idx - previous_idx <= 1:
                    current_event.append(idx)
                else:
                    loudest_frame = max(current_event, key=lambda f: onset_envelope[f])
                    detected_onset_frames.append(loudest_frame)
                    current_event = idx
                loudest_frame = max(current_event, key=lambda f: onset_envelope[f])
                detected_onset_frames.append(loudest_frame)
            detected_onset_times = librosa.frames_to_time(np.array(detected_onset_frames), sr=self.sample_rate)
        except Exception as e:
           raise e 

        # Merge near-simultaneous detections
        minimum_note_gap = 0.08 # in miliseconds
        final_onset_timestamps = []
        try:
            detected_onset_times = np.sort(detected_onset_times)
            current_onset_cluster = [detected_onset_times[0]]

            for i in range(1, len(detected_onset_times)):
                timestamp = detected_onset_times[i]
                previous_timestamp = detected_onset_times[i-1]

                if timestamp - previous_timestamp < minimum_note_gap:
                    current_onset_cluster.append(timestamp)
                else:
                    centroid_onset = np.mean(current_onset_cluster)
                    final_onset_timestamps.append(centroid_onset)
                    current_onset_cluster = timestamp
            final_onset_timestamps.append(np.mean(current_onset_cluster))
            return final_onset_timestamps
        except Exception as e:
            raise e


        
        #TODO: Finish adaptive detection
        #TODO: Experiment with the difference between rms derivative and rms delta.
