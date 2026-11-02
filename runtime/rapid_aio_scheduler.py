"""Qwen rectified-flow Euler ancestral sampling with Comfy's Beta grid.

The schedule uses Beta(0.6, 0.6) quantiles on a 10,000-point flow grid and
ComfyUI's default Qwen exponential shift (1.15). This differs from Diffusers'
``use_beta_sigmas``, which includes the final quantile and interpolates sigmas.
Equations were checked against ComfyUI's ``sample_euler_ancestral_RF``,
``beta_scheduler`` and ``ModelSamplingFlux``; no ComfyUI runtime is required.
This matches the sampling convention, not pixel-for-pixel ComfyUI output.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)
from diffusers.utils.torch_utils import randn_tensor
from scipy.stats import beta


class QwenEulerAncestralScheduler(FlowMatchEulerDiscreteScheduler):
    """Fixed-shift Beta schedule and eta=1 ancestral steps for Qwen velocity.

    Qwen's pipeline does not pass its generator into scheduler.step. A separate
    CPU noise stream therefore makes each render reproducible from its seed,
    without mutating Torch's global RNG. set_timesteps resets that stream.
    """

    @register_to_config
    def __init__(self, noise_sampler_seed: int = 0, flow_shift: float = 1.15):
        if not math.isfinite(flow_shift):
            raise ValueError("flow_shift must be finite")
        super().__init__(num_train_timesteps=1000, use_dynamic_shifting=False)
        self.num_inference_steps = None
        self._noise_generator = torch.Generator(device="cpu")

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        """Select the fixed Beta grid, accepting Qwen's pipeline call shape.

        Pipeline-provided sigmas select the step count only; their linear
        spacing and resolution-dependent mu intentionally do not replace the
        fixed Comfy Qwen schedule. Explicit custom timesteps are unsupported.
        """
        if timesteps is not None:
            raise ValueError("Rapid AIO uses a fixed Beta schedule, not custom timesteps")
        if sigmas is not None:
            if num_inference_steps is not None and len(sigmas) != num_inference_steps:
                raise ValueError("sigmas must match num_inference_steps")
            num_inference_steps = len(sigmas)
        if (
            not isinstance(num_inference_steps, int)
            or isinstance(num_inference_steps, bool)
            or not 1 <= num_inference_steps <= 1000
        ):
            raise ValueError("num_inference_steps must be an integer from 1 to 1000")

        quantiles = beta.ppf(1.0 - np.arange(num_inference_steps) / num_inference_steps, 0.6, 0.6)
        indices = np.unique(np.rint(quantiles * 9999).astype(np.int64))[::-1].copy()
        grid = torch.from_numpy(indices).to(dtype=torch.float32).add(1).div(10000)
        shift = math.exp(self.config.flow_shift)
        schedule = shift / (shift + grid.reciprocal() - 1)
        self.sigmas = torch.cat((schedule, torch.zeros(1))).to(device=device)
        self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
        self.num_inference_steps = len(self.timesteps)
        self._step_index = None
        self._begin_index = None
        self._noise_generator.manual_seed(self.config.noise_sampler_seed)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple[torch.Tensor]:
        """Integrate flow velocity, preserving x=(1-sigma)*x0+sigma*noise."""
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before step")
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()
        ):
            raise ValueError("Pass a scheduler timestep, not an integer step index")
        if self.step_index is None:
            self._init_step_index(timestep)
        sigma, target = self.sigmas[self.step_index : self.step_index + 2].to(sample.device)
        current = sample.float()
        velocity = model_output.float()
        if target == 0:
            previous = current - sigma * velocity
        else:
            # Step further down the flow, then restore target signal/noise levels.
            intermediate = target.square() / sigma
            signal_scale = (1 - target) / (1 - intermediate)
            mean = signal_scale * (current + (intermediate - sigma) * velocity)
            variance = target.square() - (signal_scale * intermediate).square()
            noise = randn_tensor(
                current.shape,
                generator=generator if generator is not None else self._noise_generator,
                device=current.device,
                dtype=current.dtype,
            )
            previous = mean + variance.clamp_min(0).sqrt() * noise
        self._step_index += 1
        previous = previous.to(model_output.dtype)
        if not return_dict:
            return (previous,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=previous)
