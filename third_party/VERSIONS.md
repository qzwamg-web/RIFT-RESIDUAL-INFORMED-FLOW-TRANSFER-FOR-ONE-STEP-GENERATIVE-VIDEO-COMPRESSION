# Third-party components

RIFT uses source components from DCVC and Wan2.2. Their licenses and notices
are included alongside the source.

- **DCVC** provides entropy models, coding layers and the rANS C++ extension.
  See [LICENSE.txt](DCVC/LICENSE.txt) and [NOTICE](<DCVC/NOTICE .txt>).
  Build the extension from `DCVC/src/cpp` with
  `python setup.py build_ext --inplace` after installing pybind11.
- **Wan2.2** provides the video DiT and VAE runtime.
  See [LICENSE.txt](Wan2.2/LICENSE.txt). Pretrained weights are external resources.

Use pretrained weights compatible with the model loader. Changes to third-party
implementations require validating checkpoint loading and numerical behavior.
