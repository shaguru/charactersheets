This is the portable worker source for Character Forge's MiniMax integration.

The Forge web interface submits jobs to `mvm forge-bridge` in this folder's separate
Python 3.12 environment. Model weights, packages, logs and generated work are ignored
by Git. Run `SETUP_MINIMAX.bat` at the repository root for optional setup.

On Tom's PC, Forge reuses the existing sibling `MINIMAX_VIDEO_Maker` environment and
cache. No model weights are copied. Keep this source in sync with changes to that
worker's bridge, image runtime, offloader and video renderer.

Model revisions and native CUDA runtime hashes are pinned in `mvm/image_native.py`.
The Diffusers commit and Python package versions match the tested local runtime.
