A moderately overengineered Python tool for onset detection and visualisation. Under the hood, it is a combination of [`librosa`](https://github.com/librosa/librosa) for the initial analysis, `scipy` for filtering, and `numpy` for additional processing and peak-picking. As with any onset detection, the output will almost certainly need manual adjustment.

## Quick Start
In Python:
```python
from onset_detection import OnsetDetect

recording = OnsetDetect("recording.wav")

recording.detect_onsets(
    envelope="hybrid",
    hybrid_env_components=["spectral_flux", "delta_rms"],
    threshold_k=1.5,
    output="list",
    plot=True
)
```
