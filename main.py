import os
#import csv
import librosa
import numpy as np
from scipy.signal import medfilt
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter, median_filter
import matplotlib.pyplot as plt


class OnsetDetect:
    def __init__(self, sound_file: str, title='', start=0, end=False) -> None:
        self.sound_file = sound_file
        self.start = start
        self.end = end
        if self.end:
            self.array, self.sample_rate = librosa.load(sound_file, offset=self.start, duration=(self.end - self.start))
        else:
            self.array, self.sample_rate = librosa.load(sound_file, offset=self.start)
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
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        self._array = value

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def start(self):
        """The start property."""
        return self._start

    @start.setter
    def start(self, value):
        if value < 0:
            raise ValueError('The value of start must be 0 or greater.')
        self._start = value

    @property
    def end(self):
        """The end property."""
        return self._end

    @end.setter
    def end(self, value):
        if value and value < 0:
            raise ValueError('The value of end must be greater than 0.')
        elif value and value < self.start:
            raise ValueError('The value of end must be greater than the value of start')
        self._end = value

    # ENVELOPES

    def _envelope_spectral_flux(self):
        spectral_flux_envelope = librosa.onset.onset_strength(
            y=self.array, sr=self.sample_rate
        )
        spectral_flux_envelope = spectral_flux_envelope / (
            np.max(spectral_flux_envelope) + 1e-6
        )
        return spectral_flux_envelope

    def _envelope_diff_spectral_flux(self):
        spectral_flux_envelope = librosa.onset.onset_strength(
            y=self.array, sr=self.sample_rate
        )
        diff_specflux_envelope = np.diff(spectral_flux_envelope)
        diff_specflux_envelope = np.maximum(0, diff_specflux_envelope)
        diff_specflux_envelope = np.concatenate([[0], diff_specflux_envelope])
        diff_specflux_envelope = diff_specflux_envelope / (
            np.max(diff_specflux_envelope) + 1e-6
        )
        return diff_specflux_envelope

    def _envelope_diff_rms(self, frame_length=2048):
        rms = librosa.feature.rms(
            y=self.array, frame_length=frame_length, hop_length=512
        )[0]
        rms_diff = np.diff(rms)
        rms_diff_rising = np.maximum(0, rms_diff)
        rms_diff_envelope = np.concatenate([[0], rms_diff_rising])
        rms_diff_envelope = rms_diff_envelope / (np.max(rms_diff_envelope) + 1e-6)
        return rms_diff_envelope

    def _envelope_delta_rms(self, frame_length=2048):
        rms = librosa.feature.rms(
            y=self.array, frame_length=frame_length, hop_length=512
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
        chroma_cqt = librosa.feature.chroma_cqt(
            y=self.array, sr=self.sample_rate, hop_length=512
        )
        chroma_cqt_diff = np.diff(chroma_cqt)
        chroma_cqt_diff = np.maximum(0, chroma_cqt_diff).mean(axis=0)
        chroma_envelope = np.concatenate([[0], chroma_cqt_diff])
        chroma_envelope = chroma_envelope / (np.max(chroma_envelope) + 1e-6)
        return (
            chroma_envelope
        )

    def _envelope_hybrid(self, *envelopes):
        for i in envelopes:
            hybrid_envelope = []

            if i == 'spectral_flux':
                envelope = np.array(self._envelope_spectral_flux())
            elif i == 'diff_spectral_flux':
                envelope = np.array(self._envelope_diff_spectral_flux())
            elif i == 'diff_rms':
                envelope = np.array(self._envelope_diff_rms())
            elif i == 'delta_rms':
                envelope = np.array(self._envelope_delta_rms())
            elif i == 'chroma_cqt':
                envelope = np.array(self._envelope_chroma_cqt())
            hybrid_envelope.append(envelope)
        hybrid_envelope = sum(hybrid_envelope)
        return hybrid_envelope

    # FILTERING
    def _median_filter(self, onset_envelope, kernel_size=3):
        onset_envelope = medfilt(onset_envelope, kernel_size=kernel_size)
        return onset_envelope

    # THRESHOLD

    def _global_mad_threshold(self, onset_envelope, k_factor=2.0):
        """Compute a moving threshold"""
        median = np.median(onset_envelope)
        mad = median_abs_deviation(onset_envelope)
        k = k_factor
        threshold = median + (k * mad)
        threshold = max(threshold, 0)
        return threshold

    def _moving_mad_threshold(self, onset_envelope, window_duration=2.0, k_factor=2.0):
        window = int(window_duration * self.sample_rate / 512)
        window_median = median_filter(onset_envelope, size=window, mode="nearest")
        window_mad = generic_filter(
            onset_envelope, median_abs_deviation, size=window, mode="nearest"
        )
        threshold = window_median + (window_mad * k_factor)
        return threshold

    # ONSET DETECTION

    def _peak_pick(self, onset_envelope, threshold):
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
                np.array(detected_onset_frames), sr=self.sample_rate
                )
        return detected_onset_times

    def _sensitive_peak_pick(self, onset_envelope, threshold):
        onset_timestamps = librosa.util.peak_pick(onset_envelope, pre_max=5, post_max=5, pre_avg=2, post_avg=2, delta=0.7, wait=2)
        onset_timestamps = librosa.frames_to_time(onset_timestamps)
        threshold = 0
        return onset_timestamps, threshold

    def _merge_onsets(self, onset_timestamps, min_note_gap=0.08):
        # Merge near-simultaneous detections
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
        onset_timestamps = np.array(onset_timestamps) + self.start
        onset_timestamps = onset_timestamps.tolist()
        if output_type == "list":
            print(onset_timestamps)
        elif output_type == "rows":
            for i in onset_timestamps:
                print(
                    f"{i},{onset_timestamps.index(i)}"
                )
        elif output_type == "csv":
            # TODO: implement csv
            pass

    # VISUALISATION

    def _plot(self, onset_timestamps, onset_envelope, threshold, threshold_type):
        times = librosa.frames_to_time(np.arange(len(onset_envelope)), sr=self.sample_rate)

        D = np.abs(librosa.stft(self.array))
        _, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            x_axis="time",
            y_axis="log",
            ax=ax[0],
            sr=self.sample_rate,
        )

        # Generate spectrogram with onset lines
        ax[0].set(title=f"{self.title}")
        ax[0].label_outer()
        ax[0].vlines(
            onset_timestamps,
            0,
            np.max(D) ** 4,
            color="g",
            alpha=0.9,
            linewidth=1,
            linestyles="solid",
            label="Onsets",
        )

        # Graph onset envelope
        ax[1].plot(times, onset_envelope, label='Onset envelope')
        ax[1].vlines(onset_timestamps, 0, np.max(onset_envelope), color='g', linewidth=2, label='Detected onsets')
        
        if threshold_type == 'global':
            ax[1].axhline(y=threshold, color='r', label='Threshold')
        elif threshold_type == 'moving':
            ax[1].plot(times, threshold, color='r', label='Threshold')


        plt.legend()
        plt.show()

    def _plot_envelope_comparison(self, times, *envelopes, spaced_view=True):
        D = np.abs(librosa.stft(self.array))
        _, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            x_axis="time",
            y_axis="log",
            ax=ax[0],
            sr=self.sample_rate,
        )

        spacing = 0
        for envelope in envelopes:
            ax[1].plot(times, spacing + envelope[0], label=envelope[1])
            if spaced_view:
                spacing += 1
        plt.legend()
        plt.show()

    def _plot_filter_comparison(self, times, *envelopes):
        # TODO: Plot every different envelope and output.
        D = np.abs(librosa.stft(self.array))
        _, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            x_axis="time",
            y_axis="log",
            ax=ax[0],
            sr=self.sample_rate,
        )

        for envelope in envelopes:
            ax[1].plot(times, 1+ envelope[0], label=envelope[1], color=envelope[2])

        for envelope in envelopes:
            ax[1].plot(times, self._median_filter(envelope[0]))

        plt.legend()
        plt.show()
        #TODO: mod 1 or remove y-axis labels in lower table.

    # PUBLIC METHODS

    def detect_onsets(self, envelope='spectral_flux', hybrid_env_components=['spectral_flux', 'delta_rms'], filtering='median_filter', filter_kernel=3, threshold_k=2.0, threshold_type='global', peak_picking='sensitive', merge_onsets=False, output='list', output_destination='onsets.csv', plot=True):
        # TODO: implement detection logic
        if envelope == 'spectral_flux':
            onset_envelope = self._envelope_spectral_flux()
        elif envelope == 'diff_spectral_flux':
            onset_envelope = self._envelope_diff_spectral_flux()
        elif envelope == 'diff_rms':
            onset_envelope = self._envelope_diff_rms()
        elif envelope == 'delta_rms':
            onset_envelope = self._envelope_delta_rms()
        elif envelope == 'chroma_cqt':
            onset_envelope = self._envelope_chroma_cqt()
        elif envelope == 'hybrid':
            onset_envelope = self._envelope_hybrid(*hybrid_env_components)
        else:
            raise ValueError(f'{envelope} is not a supported envelope type.')
        
        if filtering == 'none':
            pass
        elif filtering == 'median_filter':
            onset_envelope = self._median_filter(onset_envelope=onset_envelope, kernel_size=filter_kernel)

        if threshold_type == 'global':
            threshold = self._global_mad_threshold(onset_envelope=onset_envelope, k_factor=threshold_k)
        elif threshold_type == 'moving':
            threshold = self._moving_mad_threshold(onset_envelope=onset_envelope, k_factor=threshold_k, window_duration=10)
        else:
            raise ValueError(f'{threshold_type} is not a supported threshold type.')

        if peak_picking == 'picky':
            onset_timestamps = self._peak_pick(onset_envelope=onset_envelope, threshold=threshold)
        elif peak_picking =='sensitive':
            onset_timestamps, threshold = self._sensitive_peak_pick(onset_envelope=onset_envelope, threshold=threshold)

        if merge_onsets:
            onset_timestamps = self._merge_onsets(onset_timestamps=onset_timestamps, min_note_gap=0.08)

        self._output(output_type=output, output_destination=output_destination, onset_timestamps=onset_timestamps)

        if plot:
            self._plot(onset_timestamps=onset_timestamps, onset_envelope=onset_envelope, threshold=threshold, threshold_type=threshold_type)

    def compare(self, compare_parameter='envelopes'):
        #TODO: implement plotting
        #TODO: potential support for ground truth and statistical comparison to that?
        specflux = self._envelope_spectral_flux()
        specflux_diff = self._envelope_diff_spectral_flux()
        rms_diff = self._envelope_diff_rms()
        rms_delta = self._envelope_delta_rms()
        chroma_cqt = self._envelope_chroma_cqt()
        times = librosa.frames_to_time(np.arange(len(specflux)), sr=self.sample_rate)

        envelopes = [(specflux, 'Spectral flux', 'C0'), (specflux_diff, 'Spectral flux (derivative)', 'C1'), (rms_diff, 'Root-mean-square (derivative)', 'C2'), (rms_delta, 'Root-mean-square (delta)', 'C3'), (chroma_cqt, 'CQT chroma', 'C4')]

        if compare_parameter == 'envelopes':
            self._plot_envelope_comparison(times, *envelopes, spaced_view=False)
        elif compare_parameter == 'filtering':
            self._plot_filter_comparison(times, *envelopes)
        else:
            raise ValueError(f'{compare_parameter} is not a valid parameter to compare')


if __name__ == "__main__":
    # TODO: set default behaviour
    fp = 'data/trepak.mp3'
    OnsetDetect(fp, start=4.6, end=14.6).compare(compare_parameter='envelopes')
    OnsetDetect(fp, start=4.6, end=14.6).compare(compare_parameter='filtering')
    OnsetDetect(fp, start=4.6, end=14.6).detect_onsets(envelope='hybrid', hybrid_env_components = ['spectral_flux', 'delta_rms', 'chroma_cqt'], output='rows', threshold_k=1.5, threshold_type='moving', peak_picking='picky', merge_onsets=False)

    #TODO: replace fig with _
    #TODO: remove all individual filtering
