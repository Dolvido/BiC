"""Pixel-to-action and grounded-response model for the simulated computer.

The retinal adapter preserves spatial information before passing it to the
existing regional brain. Instructions enter through the separate comprehension
network; replies are conditioned on the completed regional computation. Goals,
correct actions, rewards, and target replies are never observation channels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ..language import ByteCodec, LanguageBrain, LanguageConfig
from ..model import Brain, BrainConfig, BrainState


@dataclass(frozen=True)
class ComputerConfig:
    hidden_size: int = 48
    language_hidden_size: int = 96
    embedding_size: int = 48
    visual_features: int = 32
    max_history: int = 3

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")


class Retina(nn.Module):
    """Small convolutional image adapter retaining a 4-by-4 spatial layout."""

    def __init__(self, visual_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(24 * 4 * 4, visual_features),
            nn.LayerNorm(visual_features),
            nn.Tanh(),
        )

    def forward(self, pixels: Tensor) -> Tensor:
        return self.network(pixels)


class ComputerBrain(nn.Module):
    """A newly initialized regional agent for a small rendered desktop.

    Eleven action outputs represent four cell clicks, field focus, the keys
    h/i/o/k, backspace, and stop. Histories contain actual previous screens and
    cursor/focus/step measurements. ``forward`` and ``respond`` reset state per
    call. Their ``with_state`` variants carry caller-owned regional activity
    across successive screen/text episodes without adding model parameters.

    All weights retain ``requires_grad=True``. Existing auxiliary classifier,
    prediction, and value heads may receive no gradient from action/reply losses;
    the trainable count therefore describes eligibility, not learned parameters.
    No older checkpoint is loaded implicitly.
    """

    NUM_ACTIONS = 11
    IMAGE_SIZE = 32

    def __init__(self, config: ComputerConfig | None = None):
        super().__init__()
        if config is not None and not isinstance(config, ComputerConfig):
            raise ValueError("config must be a ComputerConfig instance")
        self.config = config or ComputerConfig()
        self.retina = Retina(self.config.visual_features)
        self.language = LanguageBrain(
            Brain(BrainConfig(
                hidden_size=self.config.hidden_size,
                visual_dim=self.config.visual_features,
                auditory_dim=8,
                body_dim=4,
                vocab_size=8,
                num_actions=self.NUM_ACTIONS,
            )),
            LanguageConfig(
                hidden_size=self.config.language_hidden_size,
                embedding_size=self.config.embedding_size,
                max_input_bytes=128,
                max_output_bytes=64,
            ),
        )
        # Training-only perceptual supervision: four button colors (4 each),
        # visible last-click swatch (5), and visible textbox contents (21).
        # These logits never enter the action policy or response generator.
        self.scene_readout = nn.Linear(self.config.hidden_size, 42)

    @property
    def brain(self) -> Brain:
        """Expose the shared core without registering its parameters twice."""
        return self.language.brain

    def parameter_counts(self) -> dict[str, int]:
        counts = {
            "retina": sum(parameter.numel() for parameter in self.retina.parameters()),
            "scene_readout": sum(parameter.numel() for parameter in self.scene_readout.parameters()),
            **self.language.parameter_counts(),
        }
        counts["total"] = sum(counts.values())
        counts["trainable"] = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        counts["frozen"] = counts["total"] - counts["trainable"]
        return counts

    def prepare_reply(self, replies: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Return shifted teacher inputs and targets; a trainer must ignore PAD.

        Targets are supervision returned to the caller, never passed into the
        brain. At inference ``respond`` supplies its own previously generated IDs.
        """
        encoded, _ = self.language.codec.batch_encode(
            replies, max_bytes=self.language.config.max_output_bytes,
            device=next(self.parameters()).device,
        )
        return encoded[:, :-1].contiguous(), encoded[:, 1:].contiguous()

    def _validate_observations(self, pixels: Tensor, body: Tensor, prompts: Sequence[str]) -> tuple[int, int]:
        if not isinstance(pixels, Tensor) or pixels.ndim != 5 or tuple(pixels.shape[2:]) != (3, self.IMAGE_SIZE, self.IMAGE_SIZE):
            raise ValueError("pixels must have shape [batch, time, 3, 32, 32]")
        batch, time = pixels.shape[:2]
        if batch < 1 or not 1 <= time <= self.config.max_history:
            raise ValueError(f"Use a nonempty batch and a history of 1 through {self.config.max_history} frames")
        if not isinstance(body, Tensor) or body.shape != (batch, time, 4):
            raise ValueError("body must have shape [batch, time, 4]")
        parameter = next(self.parameters())
        for name, tensor in (("pixels", pixels), ("body", body)):
            if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
                raise ValueError(f"{name} must use model device {parameter.device} and dtype {parameter.dtype}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain finite values")
        if not ((pixels >= 0) & (pixels <= 1)).all():
            raise ValueError("pixels must be normalized into [0, 1]")
        if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence) or len(prompts) != batch:
            raise ValueError("prompts must contain one string per batch member")
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompts must contain only strings")
        return batch, time

    def _prepare_episode(
        self, pixels: Tensor, body: Tensor, prompts: Sequence[str],
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        batch, time = self._validate_observations(pixels, body, prompts)
        visual = self.retina(pixels.reshape(batch * time, 3, self.IMAGE_SIZE, self.IMAGE_SIZE))
        observations = {
            "visual": visual.reshape(batch, time, self.config.visual_features),
            "auditory": pixels.new_zeros(batch, time, self.brain.config.auditory_dim),
            "body": body,
            "feedback": pixels.new_zeros(batch, time, 2),
            "tokens": torch.zeros(batch, time, dtype=torch.long, device=pixels.device),
        }
        text_ids, text_lengths = self.language.prepare_inputs(prompts)
        return observations, text_ids, text_lengths

    def forward(
        self, pixels: Tensor, body: Tensor, prompts: Sequence[str], decoder_input_ids: Tensor,
        *, ablate: Iterable[str] = (),
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        observations, text_ids, text_lengths = self._prepare_episode(pixels, body, prompts)
        output = self.language(observations, text_ids, text_lengths, decoder_input_ids, ablate=ablate)
        output["scene_logits"] = self.scene_readout(output["region_activity"]["visual_cortex"][:, -1])
        return output

    def forward_with_state(
        self, pixels: Tensor, body: Tensor, prompts: Sequence[str], decoder_input_ids: Tensor,
        state: BrainState | None = None, *, ablate: Iterable[str] = (),
    ) -> tuple[dict[str, Tensor | dict[str, Tensor]], BrainState]:
        """Continue explicit visual/language episodes through the existing core.

        Pass only new frames when continuing state: replaying prior history
        advances those observations a second time. Keep batch rows assigned to
        the same sessions. The prompt is available before its supplied frames.
        State stays connected to its training graph until explicitly detached.
        Save activity with ``brain.state_to_dict`` alongside matching weights.
        """
        observations, text_ids, text_lengths = self._prepare_episode(pixels, body, prompts)
        output, next_state = self.language.forward_with_state(
            observations, text_ids, text_lengths, decoder_input_ids, state, ablate=ablate,
        )
        output["scene_logits"] = self.scene_readout(output["region_activity"]["visual_cortex"][:, -1])
        return output, next_state

    def _generate_response(self, output, prefix, max_new_bytes):
        """Decode a completed episode without running its regional core again."""
        context = output["production_context"]
        logits = output["language_logits"][:, -1]
        ended = torch.zeros(prefix.shape[0], dtype=torch.bool, device=prefix.device)
        for index in range(max_new_bytes):
            if index:
                logits = self.language.inferior_frontal(prefix, context)[:, -1]
            logits = logits.clone()
            logits[:, [ByteCodec.PAD, ByteCodec.BOS]] = -torch.inf
            selected = logits.argmax(dim=-1)
            selected = torch.where(ended, ByteCodec.PAD, selected)
            prefix = torch.cat((prefix, selected[:, None]), dim=1)
            ended = ended | (selected == ByteCodec.EOS)
            if bool(ended.all()):
                break
        return [self.language.codec.decode(row, errors="replace") for row in prefix]

    @torch.no_grad()
    def respond(
        self, pixels: Tensor, body: Tensor, prompts: Sequence[str], *, max_new_bytes: int = 48,
        ablate: Iterable[str] = (),
    ) -> list[str]:
        """Greedily generate bytes, stopping on EOS or the explicit byte budget.

        The regional episode is computed once. Each successive byte conditions
        only on its generated prefix and that episode's production context. PAD
        and BOS cannot be emitted. Invalid/incomplete UTF-8 is displayed with
        replacement characters explicitly; random initial weights are not fluent.
        """
        if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= self.language.config.max_output_bytes:
            raise ValueError(f"max_new_bytes must be in [1, {self.language.config.max_output_bytes}]")
        batch, _ = self._validate_observations(pixels, body, prompts)
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            prefix = torch.full((batch, 1), ByteCodec.BOS, dtype=torch.long, device=pixels.device)
            output = self(pixels, body, prompts, prefix, ablate=ablate)
            return self._generate_response(output, prefix, max_new_bytes)
        finally:
            # Preserve individually frozen/evaluation submodule modes too.
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def respond_with_state(
        self, pixels: Tensor, body: Tensor, prompts: Sequence[str], state: BrainState | None = None,
        *, max_new_bytes: int = 48, ablate: Iterable[str] = (),
    ) -> tuple[list[str], BrainState]:
        """Generate a reply and advance regional state exactly once per episode.

        Response bytes are greedily generated from BOS using the completed
        episode's production context. Generated bytes do not become recurrent
        brain inputs. Returned activity is for inference; use forward_with_state
        when a training graph must span screen/text episodes.
        """
        if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= self.language.config.max_output_bytes:
            raise ValueError(f"max_new_bytes must be in [1, {self.language.config.max_output_bytes}]")
        batch, _ = self._validate_observations(pixels, body, prompts)
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            prefix = torch.full((batch, 1), ByteCodec.BOS, dtype=torch.long, device=pixels.device)
            output, next_state = self.forward_with_state(pixels, body, prompts, prefix, state, ablate=ablate)
            return self._generate_response(output, prefix, max_new_bytes), next_state
        finally:
            for module, mode in modes:
                module.training = mode
