import numpy as np
import librosa
import os
import matplotlib.pyplot as plt


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

    def _preprocess(self, threshold, onset_envelope):
        onset_frames = librosa.util.peak_pick(
                onset_envelope,
                pre_avg=3,
                post_avg=3,
                pre_max=3,
                post_max=3,
                wait=3,
                delta=0,
                )

        median_floor = np.median(onset_envelope)
        calculated_threshold = threshold * median_floor

        picked_peaks = []
        for frame in onset_frames:
            if onset_envelope[frame] > calculated_threshold:
                picked_peaks.append(frame)
        picked_peaks = np.array(picked_peaks)

        return picked_peaks, onset_frames, calculated_threshold

    def compute_peak_onset(
        self, preemphasis_coefficient=0.97, threshold=False, plot=False
    ) -> list:
        self.array = librosa.effects.preemphasis(
            self.array, coef=preemphasis_coefficient
        )
        onset_envelope = librosa.onset.onset_strength(y=self.array, sr=self.sample_rate)

        if threshold:
            onset_frames, raw_onset_frames, calculated_threshold = self._preprocess(
                threshold=threshold, onset_envelope=onset_envelope
            )
            #times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            times_in_envelope = librosa.frames_to_time(np.arange(len(onset_envelope)), sr=self.sample_rate)
            times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
        else:
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_envelope, sr=self.sample_rate
            )
            times = librosa.times_like(onset_envelope, sr=self.sample_rate)
            times = times[onset_frames]
        #print(times, len(onset_envelope[onset_frames]), len(times))

        if plot:
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
                linestyles="solid",
                label="Onsets",
            )
            ax[1].plot(times_in_envelope, onset_envelope, label='Raw Onset Envelope', alpha=0.6)
            plt.axhline(y=calculated_threshold, color='r', linestyle='--', label=f'Threshold')
            plt.vlines(times, 0, np.max(onset_envelope), colors='green', linestyles='solid', linewidth=2, label='Valid Onsets')
            rejected_frames = [f for f in raw_onset_frames if f not in onset_frames]
            if len(rejected_frames) > 0:
                plt.vlines(times_in_envelope[rejected_frames], 0, np.max(onset_envelope), colors='orange', linestyles='dotted', linewidth=1, label='Rejected (Below Threshold)')
            ax[1].legend()
            # ax[1].legend()
            plt.show()
        return times

    def compute_backtrack_onset(self, plot=False) -> list:
        onset_envelope = librosa.onset.onset_strength(y=self.array, sr=self.sample_rate)
        onset_raw = librosa.onset.onset_detect(
            onset_envelope=onset_envelope, backtrack=False
        )
        onset_backtrack = librosa.onset.onset_backtrack(onset_raw, onset_envelope)
        print(onset_backtrack.tolist())

        if plot:
            S = np.abs(librosa.stft(y=self.array))
            rms = librosa.feature.rms(S=S)
            times = librosa.times_like(onset_envelope, sr=self.sample_rate)
            onset_backtrack_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])

            fig, ax = plt.subplots(nrows=2, sharex=True)
            librosa.display.specshow(
                librosa.amplitude_to_db(S, ref=np.max),
                x_axis="time",
                y_axis="log",
                ax=ax[0],
            )
            ax[0].set(title=f"Backtrack onset: {self.sound_file}")
            ax[0].label_outer()
            ax[1].plot(times, rms[0], label="RMS")
            # ax[0].vlines(librosa.frames_to_time(onset_backtrack_rms), 0, rms.max(), label='Backtracked RMS', color='g', linestyle='--')
            ax[0].vlines(
                librosa.frames_to_time(onset_backtrack_rms),
                0,
                8192,
                label="Backtracked RMS",
                color="g",
                linestyle="--",
            )
            plt.show()

    # TODO: Onset detection, backtrack
    # TODO: Score file integration
    # TODO: Export options
    # TODO: Consolidate plotting into a single method.
    # TODO: Is maybe backtrack based on spectral flux more appropriate here than RMS?
    # Another idea. Maybe I should run it through the mel filter bank first. Allegedly that shoud bias the frequency bands biased by the human ear.


if __name__ == "__main__":
    #print(OnsetDetect('data/rytel-A1.wav').compute_backtrack_onset(plot=True))
    onset = (OnsetDetect("data/rytel-A1.wav").compute_peak_onset(preemphasis_coefficient=0.97, threshold=1.2, plot=True))
    for i in onset:
        print(f"{i},peak_preprocessed")
