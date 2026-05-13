import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import median_abs_deviation


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

    def _diff_preprocess(self, threshold, onset_envelope):
        diff_onset_envelope = np.diff(onset_envelope)
        diff_onset_envelope = np.maximum(0, diff_onset_envelope)
        diff_onset_envelope = np.concatenate([[0], diff_onset_envelope])
        smooth_diff_onset_env = medfilt(diff_onset_envelope, kernel_size=3)

        threshold_value = np.percentile(smooth_diff_onset_env, 75)
        onset_frames_above_threshold = np.where(smooth_diff_onset_env > threshold_value)[0]

        onset_frames = []
        if len(onset_frames_above_threshold) > 0:
            #TODO add remaining logic here. I think this is the right way to go.

            return smooth_diff_onset_env

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

    def adaptive_detection(self, k=2.0, frame_length=2048):
        # Compute two envelopes: spectral flux and root-mean-square derivative.
        spectral_flux_envelope = librosa.onset.onset_strength(y=self.array, sr=self.sample_rate)
        rms = librosa.feature.rms(y=self.array, frame_length=frame_length, hop_length=512)[0]
        rms_diff = np.diff(rms)
        rms_diff_rising = np.maximum(0, rms_diff)
        rms_diff_envelope = np.concatenate([[0], rms_diff_rising])

        # Normalise both envelopes with epsilon smoothing to avoid div by 0.
        spectral_flux_envelope = spectral_flux_envelope / (np.max(spectral_flux_envelope) + 1e-6)
        rms_diff_envelope = rms_diff_envelope / (np.max(rms_diff_envelope) + 1e-6)

        # Add complimentary envelopes
        hybrid_envelope = spectral_flux_envelope + rms_diff_envelope
        
        # Smooth envelope
        smoothed_hybrid_envelope = medfilt(hybrid_envelope, kernel_size=6)

        # Compute an adaptive threshold
        median = np.median(smoothed_hybrid_envelope)
        mad = median_abs_deviation(smoothed_hybrid_envelope)
        k_factor = k
        threshold = median + (k_factor * mad)
        threshold = max(threshold, 0)
        
        #TODO: Finish adaptive detection
        #TODO: Experiment with the difference between rms derivative and rms delta.
