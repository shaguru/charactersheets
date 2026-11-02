# Character Forge

Generate character sheets and scenes locally with Qwen Rapid AIO v19, reference images, and a browser studio. The app runs directly on PyTorch CUDA and Diffusers.

## Quick start for Windows

```bat
git clone https://github.com/shaguru/charactersheets.git
cd charactersheets
INSTALL.bat
```

Or download the repository ZIP, extract it, and double-click **INSTALL.bat**. GitHub access to this private repository is required. Keep the files together in a writable local folder.

One click performs the whole setup:

1. Checks the NVIDIA GPU, driver, RAM and available disk space.
2. Downloads a checksum-verified Python 3.12.10 distribution into this folder and creates `runtime/.venv`.
3. Installs the pinned Python packages, including PyTorch with CUDA 12.8. No separate CUDA Toolkit, Python installation or administrator rights are needed.
4. Runs an actual BF16 CUDA calculation to verify GPU compatibility.
5. Downloads the Qwen Rapid AIO v19 checkpoint and its tokenizer/configuration files. New downloads are checked against the catalog sizes and SHA-256 hashes.
6. Starts the server and opens **http://127.0.0.1:8288/forge/** in your browser.

First setup downloads roughly **33 GB**, mostly the 28.4 GB model. Allow time for downloads, installation and checksum verification. Leave the console window open while using the app. Close it or press Ctrl+C to stop.

After installation, double-click **START.bat**. It checks the local environment and reuses existing packages and models. An interrupted model download resumes when you run it again. **INSTALL.bat** is also safe to rerun. **SETUP.bat** installs/checks everything without starting the server.

## Hardware

- Windows 10/11, x64.
- NVIDIA Ampere or newer GPU with native BF16 support, such as RTX 30/40/50 series. AMD, Intel and CPU generation are not supported.
- NVIDIA driver **570.65 or newer**. If it is missing or too old, setup explains how to update it; the display driver itself requires installation through NVIDIA.
- **24 GB VRAM and 64 GB system RAM recommended.** This configuration was verified on an RTX 3090 Ti. Smaller GPUs may need smaller images and can run out of memory. Qwen uses sequential CPU offload.
- Allow **45 GB free** for first setup. Optional SDXL models need another 11 GB or more. Use a local drive with room for generated images; an SSD is preferable.
- Internet for setup and model downloads. Rendering uses local weights and configuration files.

## Use the studio

- Drop or paste a character reference. Qwen accepts up to three references for character, outfit or location.
- Pick a sheet layout, describe a scene, or write a free prompt. Layout presets are prompt templates; exact poses and panel arrangements can vary.
- Use dynamic choices such as `[small | big | long]` to vary each generated image. This also works in the negative prompt.
- Qwen defaults to four steps and guidance 1.0. Set guidance above 1.0 to enable negative prompting.
- Pick a batch size and queue. Images render one at a time; the model stays cached for later images.
- Click a gallery image to enlarge it. **Click the enlarged image again to close**, click the surrounding background, or press Escape. Arrow keys move between images.
- Favourite, download, reuse a reference, load saved settings, or select images to delete. The current project's entire gallery can also be deleted.
- Use **Unload model** to release cached RAM and VRAM after a batch. **Stop all** cancels queued work and interrupts the current render at its next cancellation checkpoint.

## Optional SDXL model

Qwen is the default and the only model downloaded by the one-click installer. For the lighter photographic alternative, download **CyberRealistic XL v10** from the Models tab. Its shared image encoder, face adapter and local configuration are prepared automatically. SDXL uses the first reference only and preserves clothing less reliably than Qwen.

To install both models from a terminal:

```bat
SETUP.bat -Models qwen-rapid-aio,cyberrealistic-xl
START.bat
```

For dependency installation without model downloads: `SETUP.bat -EngineOnly`.

The default Qwen checkpoint is the publisher's **Rapid AIO NSFW v19** build. Review the [publisher's model card and terms](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO) and each optional model's license before use or redistribution. Weights are downloaded from their publishers and are not included in this repository.

## Local files

| Folder | Contents |
|---|---|
| `runtime/web/` | Browser studio |
| `runtime/direct_server.py` | Local server, render queue and worker |
| `runtime/presets.json` | Editable prompt/layout presets |
| `runtime/models.json` | Model download URLs, sizes and checksums |
| `runtime/python312/`, `runtime/.venv/` | Local Python and dependencies |
| `data/models/` | Downloaded model weights and configuration |
| `data/input/forge_refs/` | Uploaded reference images |
| `data/output/<Project>/` | Generated images, generation journal and favourites |
| `data/user/` | Local preferences and optional Hugging Face token |
| `data/cache/`, `data/logs/` | Download caches and installer diagnostics |

Personal images, settings, tokens, caches, models and installed environments are excluded from Git. A fresh clone opens with an empty gallery. Legacy local ComfyUI folders are not used or shared.

## Troubleshooting

- **Setup failed:** read the console and `data/logs/setup.log`, fix the reported issue, and rerun START.bat. Completed downloads are reused.
- **CUDA check failed:** update the NVIDIA driver and restart Windows. Setup checks real GPU execution and does not switch silently to CPU.
- **Out of memory:** close other GPU applications, reduce output dimensions, and use one model at a time. The model needs substantial system RAM as well as VRAM.
- **Missing references after clearing images:** remove the old reference tiles and upload a reference again.
- **Folder moved:** START.bat repairs the local interpreter path. Moving between PCs still requires a compatible NVIDIA GPU and driver.
- **Port in use:** close the other app, or run `set FORGE_PORT=8290` in Command Prompt before running START.bat.
- **Python cannot start:** install the [Microsoft Visual C++ x64 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), then retry.

The server listens only on `127.0.0.1`. Interrupted renders remain marked as interrupted after a restart; they do not restart silently.

## Developer checks

```bat
runtime\.venv\Scripts\python.exe -m unittest discover -s runtime -p test_rapid_aio.py -v
cd runtime\qa
npm install
npm test
```

The optional browser check requires Node.js and Microsoft Edge. It uses synthetic gallery entries and does not create images or modify projects. See `runtime/qa/README.md`.
