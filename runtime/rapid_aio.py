"""Offline loading of Phr00t's original v19 Comfy AIO checkpoint.

All three sets of weights come from the selected checkpoint. Only architecture,
tokenization and processor configuration come from the pinned Qwen release.
The state mapping is validated against every checkpoint key, shape and dtype
before loading into Diffusers 0.40 / Transformers 5.15.1 modules.
"""
from __future__ import annotations

import json
import mmap
import struct
import warnings
from pathlib import Path

import torch


class _TensorInfo:
    def __init__(self, spec):
        self.shape = tuple(spec["shape"])
        self.dtype = spec["dtype"]
        self.start, self.end = spec["data_offsets"]

    def get_shape(self):
        return list(self.shape)

    def get_dtype(self):
        return self.dtype


class _SafeTensorReader:
    """Read-only storage mapping, avoiding Windows' large copy-on-write commit.

    FP8 parameters are only moved or cast during inference, never changed in
    place. Torch tensors retain a reference to their backing mapping.
    """
    _DTYPES = {
        "F8_E4M3": torch.float8_e4m3fn,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "I64": torch.int64,
        "I32": torch.int32,
        "U8": torch.uint8,
    }

    def __init__(self, path):
        self.file = open(path, "rb")
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        if not 0 < header_size <= 100_000_000:
            self.file.close()
            raise ValueError("Invalid Rapid AIO safetensors header size")
        header = json.loads(self.file.read(header_size))
        self.data_start = 8 + header_size
        self.data_size = Path(path).stat().st_size - self.data_start
        self.info = {k: _TensorInfo(v) for k, v in header.items() if k != "__metadata__"}
        self.buffer = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return self.info.keys()

    def get_slice(self, name):
        return self.info[name]

    def get_tensor(self, name):
        info = self.info[name]
        if not 0 <= info.start <= info.end <= self.data_size:
            raise ValueError(f"Invalid tensor offsets: {name}")
        if info.start == info.end:
            return torch.empty(info.shape, dtype=self._DTYPES[info.dtype])
        dtype = self._DTYPES[info.dtype]
        count = (info.end - info.start) // torch.empty((), dtype=dtype).element_size()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            return torch.frombuffer(self.buffer, dtype=dtype, count=count,
                                    offset=self.data_start + info.start).reshape(info.shape)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()


def _component_keys(keys) -> dict[str, dict[str, str]]:
    """Map destination names to source names, preserving every inference weight."""
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    result = {"transformer": {}, "text_encoder": {}, "vae": {}}
    ignored = {
        "model.diffusion_model.__index_timestep_zero__",
        "text_encoders.qwen25_7b.logit_scale",
    }
    for source in keys:
        if source in ignored:
            continue
        if source.startswith("model.diffusion_model."):
            component, name = "transformer", source.removeprefix("model.diffusion_model.")
        elif source.startswith("vae."):
            component, name = "vae", source.removeprefix("vae.")
        elif source.startswith("text_encoders.qwen25_7b.transformer.model."):
            component = "text_encoder"
            name = "language_model." + source.removeprefix("text_encoders.qwen25_7b.transformer.model.")
        elif source.startswith("text_encoders.qwen25_7b.transformer.visual."):
            component = "text_encoder"
            name = "visual." + source.removeprefix("text_encoders.qwen25_7b.transformer.visual.")
        else:
            raise ValueError(f"Unsupported Rapid AIO checkpoint key: {source}")
        if name in result[component]:
            raise ValueError(f"Duplicate Rapid AIO {component} key: {name}")
        result[component][name] = source
    # Qwen's VAE retains Wan's original naming. This converter only renames
    # entries, so strings can be mapped and validated before reading any tensor.
    original_vae = result["vae"]
    result["vae"] = convert_wan_vae_to_diffusers(original_vae)
    if len(result["vae"]) != len(original_vae):
        raise ValueError("Duplicate Rapid AIO VAE keys after conversion.")
    return result


def _validate_state(model, mapping, checkpoint, component: str) -> None:
    """Reject partial, foreign and shape-incompatible weights before allocation."""
    expected = model.state_dict()
    missing, extra = set(expected) - set(mapping), set(mapping) - set(expected)
    if missing or extra:
        raise ValueError(
            f"Invalid Rapid AIO {component} keys: missing={sorted(missing)[:4]}, "
            f"unexpected={sorted(extra)[:4]}"
        )
    for name, tensor in expected.items():
        source = mapping[name]
        stored = checkpoint.get_slice(source)
        if tuple(stored.get_shape()) != tuple(tensor.shape):
            raise ValueError(f"Invalid Rapid AIO {component} shape: {source}")
        wanted_dtype = "BF16" if component == "vae" else "F8_E4M3"
        if stored.get_dtype() != wanted_dtype:
            raise ValueError(f"Invalid Rapid AIO {component} dtype: {source}")


def _load_state(model, mapping, checkpoint, *, fp8: bool, progress=None) -> None:
    """Install one tensor at a time, keeping Linear weights in original FP8."""
    import torch
    from diffusers.hooks import apply_layerwise_casting

    for index, (name, source) in enumerate(mapping.items()):
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        value = checkpoint.get_tensor(source)
        # Norms, embeddings and convolution kernels compute in BF16. Keeping
        # their parameters BF16 also makes Transformers report the compute dtype.
        dtype = torch.float8_e4m3fn if fp8 and isinstance(parent, torch.nn.Linear) else torch.bfloat16
        value = value.to(dtype=dtype)
        if attribute in parent._parameters:
            setattr(parent, attribute, torch.nn.Parameter(value, requires_grad=False))
        else:
            parent.register_buffer(attribute, value)
        if progress and (index % 25 == 0 or index + 1 == len(mapping)):
            progress(index + 1, len(mapping))
    if fp8:
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                apply_layerwise_casting(
                    layer, storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=torch.bfloat16, skip_modules_pattern=None,
                )
    model.eval().requires_grad_(False)


def load_rapid_aio(source_path, config_dir, *, reference: bool, progress=None):
    """Build one offline portrait or reference pipeline; caller owns offloading.

    Reload when changing pipeline mode. Sharing active pipelines would let their
    Accelerate offload hooks interfere. No training or stock-weight substitution
    occurs, and no inference request can trigger a download.
    """
    from accelerate import init_empty_weights
    from diffusers import (
        AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
        QwenImageEditPlusPipeline, QwenImagePipeline, QwenImageTransformer2DModel,
    )
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2Tokenizer, Qwen2VLProcessor

    config_dir = Path(config_dir)
    def read_config(component):
        return json.loads((config_dir / component / "config.json").read_text(encoding="utf-8"))

    transformer_config = read_config("transformer")
    if transformer_config.get("zero_cond_t") is not True:
        raise ValueError("Rapid AIO v19 requires the Qwen 2511 zero_cond_t configuration.")
    with init_empty_weights():
        models = {
            "transformer": QwenImageTransformer2DModel.from_config(transformer_config),
            # The checkpoint contains the multimodal backbone, without a
            # language-generation head. Both pipelines consume hidden states.
            "text_encoder": Qwen2_5_VLModel(Qwen2_5_VLConfig.from_dict(read_config("text_encoder"))),
            "vae": AutoencoderKLQwenImage.from_config(read_config("vae")),
        }
    with _SafeTensorReader(source_path) as checkpoint:
        mappings = _component_keys(checkpoint.keys())
        for marker, shape, dtype in (
            ("model.diffusion_model.__index_timestep_zero__", (0,), "F32"),
            ("text_encoders.qwen25_7b.logit_scale", (), "F8_E4M3"),
        ):
            if marker not in checkpoint.keys():
                raise ValueError(f"Missing Rapid AIO v19 marker: {marker}")
            stored = checkpoint.get_slice(marker)
            if tuple(stored.get_shape()) != shape or stored.get_dtype() != dtype:
                raise ValueError(f"Invalid Rapid AIO v19 marker: {marker}")
        for component, model in models.items():
            _validate_state(model, mappings[component], checkpoint, component)
        for component, model in models.items():
            def report(done, total, component=component):
                if progress:
                    progress(component, done, total)
            _load_state(model, mappings[component], checkpoint, fp8=component != "vae", progress=report)
    components = dict(
        **models,
        tokenizer=Qwen2Tokenizer.from_pretrained(config_dir / "tokenizer", local_files_only=True),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            config_dir / "scheduler", local_files_only=True,
        ),
    )
    if reference:
        components["processor"] = Qwen2VLProcessor.from_pretrained(
            config_dir / "processor", local_files_only=True,
        )
        return QwenImageEditPlusPipeline(**components)
    return QwenImagePipeline(**components)
