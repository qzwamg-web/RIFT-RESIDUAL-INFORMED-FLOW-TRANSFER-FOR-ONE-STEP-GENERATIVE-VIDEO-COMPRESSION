# Vendored runtime sources

Only the DCVC and Wan2.2 source trees needed by the C512 runtime are included.
Their original licenses/notices remain beside the source. File hashes are in
RELEASE_MANIFEST.json; this is a frozen local snapshot, not a claim that the
upstream HEAD is identical. Prebuilt extensions, build folders, model weights
and unrelated baseline implementations are excluded.

- DCVC: Microsoft DCVC runtime and rANS C++ extension; see DCVC/LICENSE.txt and
  DCVC/NOTICE .txt. Build with `python setup.py build_ext --inplace` from
  DCVC/src/cpp after installing pybind11. Only Python package metadata was aligned
  to the project's3.10/3.11 environment; C++ source remains unchanged.
- Wan2.2: pretrained Wan runtime; see Wan2.2/LICENSE.txt. RIFT calls its48-channel
  TI2V backbone and VAE through raqvc/backbone; pretrained weights stay external.

Do not replace either snapshot or a pretrained base when reproducing the C512
numerical endpoint without revalidating checkpoint loading and output parity.
