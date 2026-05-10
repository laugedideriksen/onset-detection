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

    @property
    def sample_rate(self):
        return self._sample_rate

    def compute_peak_onset(self, plot=False) -> list:
        onset_array = librosa.onset.onset_detect(
            y=self.array, sr=self.sample_rate, units="time"
        )

        if plot:
            cqt = np.abs(librosa.cqt(y=self.array, sr=self.sample_rate))
            onset_envelope = librosa.onset.onset_strength(
                sr=self.sample_rate, S=librosa.amplitude_to_db(cqt, ref=np.max)
            )
            times = librosa.times_like(onset_envelope, sr=self.sample_rate)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_envelope, sr=self.sample_rate
            )

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
            ax[1].plot(times, onset_envelope, label="Onset strength")
            # ax[0].vlines(times[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
            ax[0].vlines(
                times[onset_frames],
                0,
                8192,
                color="r",
                alpha=0.9,
                linestyle="--",
                label="Onsets",
            )
            # ax[1].legend()
            plt.show()
        return onset_array.tolist()

    def compute_backtrack_onset(self, plot=False) -> list:
        cqt = np.abs(librosa.cqt(y=self.array, sr=self.sample_rate))
        onset_envelope = librosa.onset.onset_strength(
            sr=self.sample_rate, S=librosa.amplitude_to_db(cqt, ref=np.max)
        )  # compute envelope useing a constant Q spectrogram (lograrithmic, not linear)
        onset_raw = librosa.onset.onset_detect(
            onset_envelope=onset_envelope, backtrack=False
        )
        onset_backtrack = librosa.onset.onset_backtrack(onset_raw, onset_envelope)
        print(self.sample_rate)

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
            ax[0].vlines(
                librosa.frames_to_time(onset_backtrack),
                0,
                8192,
                label="Backtracked RMS",
                color="g",
                linestyle="--",
            )
            plt.show()
        return librosa.frames_to_time(onset_backtrack)


if __name__ == "__main__":
    # print(OnsetDetect('data/rytel-A1.wav').compute_backtrack_onset(plot=True))
    onset = OnsetDetect("data/rytel-A1.wav").compute_peak_onset(plot=True)
    onset
    for i in onset:
        print(f"{i},peak")
