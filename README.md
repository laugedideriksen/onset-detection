A moderately overengineered Python tool for onset detection and visualisation of noisy recordings of solo instruments. Under the hood, it is a combination of [`librosa`](https://github.com/librosa/librosa) for the initial analysis, `scipy` for filtering, and `numpy` for additional processing and peak-picking.

## Quick Start
```python
import onset_detect as od

recording = od.OnsetDetect("recording.wav")

recording.detect_onsets(
    envelope="hybrid",
    hybrid_env_components=["spectral_flux", "delta_rms"],
    threshold_k=1.5,
    output="list",
    plot=True
)
```
