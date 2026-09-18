"""Persistent visual evidence used by BiC's existing regional action network.

The explicit name parser addresses memory only. It never parses a direction or
selects a candidate. All four visual similarities enter a learned hippocampal
pathway, and the existing motor cortex produces the final eleven action logits.
The four glyph crops occupy a fixed 2x2 grid, supplied by the environment.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import torch
from torch import Tensor, nn

from .associative import AssociativeMemory, EncoderConfig, GlyphEncoder, encoder_hash
from .computer_use.model import ComputerBrain, ComputerConfig
from .language import ByteCodec

SCHEMA = "bic-regional-memory-v1"
QUOTED_NAME = re.compile(r'"(?:[^"\\]|\\.)*"')


@dataclass(frozen=True)
class MemoryControlConfig:
    steps: int = 2

    def __post_init__(self):
        if type(self.steps) is not int or not 1 <= self.steps <= 8:
            raise ValueError("steps must be an integer from 1 through 8")


def normalize_instruction(instruction: str) -> tuple[str, str]:
    """Resolve one JSON-quoted symbolic label; leave instruction semantics learned.

    Masking the name makes the parser's contribution explicit: novel labels are
    memory addresses, not words acquired by the neural comprehension network.
    JSON escaping permits names containing quotation marks and backslashes.
    """
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 512:
        raise ValueError('Enter an instruction with one quoted name, such as find "dax"')
    matches = list(QUOTED_NAME.finditer(instruction))
    if len(matches) != 1:
        raise ValueError("Use exactly one JSON-quoted object name")
    match = matches[0]
    outside = instruction[:match.start()] + " <name> " + instruction[match.end():]
    if '"' in outside:
        raise ValueError("Unbalanced quotation marks in instruction")
    try:
        label = AssociativeMemory._label(json.loads(match.group()))
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError("The quoted name must be a valid visible JSON string") from error
    normalized = " ".join(outside.lower().split())
    ByteCodec(128).encode(normalized)
    return label, normalized


class RegionalMemoryAgent(nn.Module):
    """Shared computer/language weights plus a small persistent-recall adapter.

    ``forward_evidence`` is also the training interface for cached frozen visual
    embeddings. Evidence contains no chosen index, threshold decision, direction
    label, or target action. Availability is a legitimate result of name lookup.
    ``blank_memory`` removes the complete external recall after projection.
    """

    def __init__(self, computer: ComputerBrain, encoder: GlyphEncoder,
                 config: MemoryControlConfig | None = None):
        super().__init__()
        if not isinstance(computer, ComputerBrain) or not isinstance(encoder, GlyphEncoder):
            raise TypeError("Regional control requires a ComputerBrain and GlyphEncoder")
        if (next(computer.parameters()).dtype != torch.float32
                or next(encoder.parameters()).dtype != torch.float32):
            raise ValueError("Regional memory checkpoints currently require float32 networks")
        self.config = config or MemoryControlConfig()
        if not isinstance(self.config, MemoryControlConfig):
            raise TypeError("config must be a MemoryControlConfig")
        self.computer = computer
        self.encoder = encoder.eval().requires_grad_(False)
        self.encoder_identity = encoder_hash(encoder)
        h = computer.config.hidden_size
        parameter = next(computer.parameters())
        self.recall_adapter = nn.Sequential(nn.Linear(5, h), nn.Tanh()).to(
            device=parameter.device, dtype=parameter.dtype)
        self.encoder.to(device=parameter.device, dtype=parameter.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward_evidence(self, scores: Tensor, known: Tensor, prompts: Sequence[str],
                         decoder_input_ids: Tensor | None = None, *,
                         ablate: Iterable[str] = (), blank_memory: bool = False):
        parameter = next(self.computer.parameters())
        if not isinstance(scores, Tensor) or scores.ndim != 2 or scores.shape[1] != 4 or len(scores) < 1:
            raise ValueError("scores must have shape [batch,4]")
        batch = len(scores)
        if not isinstance(known, Tensor) or known.shape != (batch, 1):
            raise ValueError("known must have shape [batch,1]")
        for name, tensor in (("scores", scores), ("known", known)):
            if tensor.device != parameter.device or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite and match the computer device/dtype")
        if ((scores < -1) | (scores > 1)).any() or ((known != 0) & (known != 1)).any():
            raise ValueError("scores must be cosine similarities and known must be 0 or 1")
        if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence) or len(prompts) != batch:
            raise ValueError("prompts must contain one normalized instruction per example")
        if type(blank_memory) is not bool:
            raise ValueError("blank_memory must be a boolean")
        steps = self.config.steps
        # Unknown names carry no visual referent evidence. No acceptance rule or
        # argmax is applied to known names before the learned regional policy.
        recall = self.recall_adapter(torch.cat((scores * known, known), dim=-1))
        if blank_memory:
            recall = torch.zeros_like(recall)
        c = self.computer.brain.config
        observations = {
            "visual": scores.new_zeros(batch, steps, c.visual_dim),
            "auditory": scores.new_zeros(batch, steps, c.auditory_dim),
            "body": scores.new_zeros(batch, steps, c.body_dim),
            "feedback": scores.new_zeros(batch, steps, 2),
            "tokens": torch.zeros(batch, steps, dtype=torch.long, device=scores.device),
        }
        language = self.computer.language
        ids, lengths = language.prepare_inputs(prompts)
        if decoder_input_ids is None:
            decoder_input_ids = torch.full((batch, 1), ByteCodec.BOS, dtype=torch.long, device=scores.device)
        return language(observations, ids, lengths, decoder_input_ids, ablate=ablate,
                        memory_context=recall[:, None].expand(-1, steps, -1))

    def forward(self, *args, **kwargs):
        return self.forward_evidence(*args, **kwargs)

    def prepare_evidence(self, memory: AssociativeMemory, label: str,
                         candidate_pixels: Tensor | list[Tensor]) -> tuple[Tensor, Tensor]:
        if not isinstance(memory, AssociativeMemory):
            raise TypeError("memory must be an AssociativeMemory")
        if memory.model_hash != self.encoder_identity or encoder_hash(self.encoder) != self.encoder_identity:
            raise ValueError("Regional controller and memory must use the same frozen visual encoder")
        evidence = memory.evidence(label, candidate_pixels)
        if evidence["scores"].shape != (4,):
            raise ValueError("Regional control requires four crops in 2x2 row-major order")
        parameter = next(self.computer.parameters())
        scores = evidence["scores"].unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
        known = scores.new_tensor([[float(evidence["known"])]])
        return scores, known

    def _decode(self, output, max_new_bytes=32):
        language = self.computer.language
        if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= language.config.max_output_bytes:
            raise ValueError("max_new_bytes must fit the language output budget")
        context = output["production_context"]
        prefix = torch.full((len(context), 1), ByteCodec.BOS, dtype=torch.long, device=context.device)
        ended = torch.zeros(len(context), dtype=torch.bool, device=context.device)
        for _ in range(max_new_bytes):
            logits = language.inferior_frontal(prefix, context)[:, -1].clone()
            logits[:, [ByteCodec.PAD, ByteCodec.BOS]] = -torch.inf
            token = logits.argmax(-1)
            token = torch.where(ended, ByteCodec.PAD, token)
            prefix = torch.cat((prefix, token[:, None]), dim=1)
            ended |= token == ByteCodec.EOS
            if bool(ended.all()):
                break
        return [language.codec.decode(row, errors="replace") for row in prefix]

    @torch.no_grad()
    def respond_evidence(self, scores, known, prompts, max_new_bytes=32, *, ablate=(), blank_memory=False):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            output = self.forward_evidence(scores, known, prompts, ablate=ablate, blank_memory=blank_memory)
            return self._decode(output, max_new_bytes)
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def act(self, memory: AssociativeMemory, instruction: str, candidate_pixels: Tensor | list[Tensor]) -> dict:
        label, normalized = normalize_instruction(instruction)
        scores, known = self.prepare_evidence(memory, label, candidate_pixels)
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            output = self.forward_evidence(scores, known, [normalized])
            action = int(output["logits"][0, -1].argmax())
            reply = self._decode(output)[0]
            return {"action": action, "index": action if 0 <= action < 4 else None,
                    "stopped": action == 10, "invalid_grid_action": 4 <= action < 10,
                    "reply": reply, "action_source": "regional_motor_cortex",
                    "response_source": "neural_byte_decoder", "scores": scores[0].tolist(),
                    "known_label": bool(known.item()), "label": label,
                    "normalized_instruction": normalized,
                    "region_activity": {name: float(value[0, -1].norm())
                                        for name, value in output["region_activity"].items()}}
        finally:
            for module, mode in modes:
                module.training = mode


def save_regional_checkpoint(path: str | Path, agent: RegionalMemoryAgent, metadata: dict | None = None):
    if encoder_hash(agent.encoder) != agent.encoder_identity:
        raise ValueError("Visual encoder changed during regional training")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema": SCHEMA, "computer_config": asdict(agent.computer.config),
                  "encoder_config": asdict(agent.encoder.config), "config": asdict(agent.config),
                  "encoder_hash": agent.encoder_identity,
                  "model": {name: value.detach().cpu() for name, value in agent.state_dict().items()},
                  "metadata": metadata or {}}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_regional_checkpoint(path: str | Path, device: str = "cpu") -> RegionalMemoryAgent:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA:
        raise ValueError("Unsupported regional memory checkpoint")
    agent = RegionalMemoryAgent(ComputerBrain(ComputerConfig(**checkpoint["computer_config"])),
                                GlyphEncoder(EncoderConfig(**checkpoint["encoder_config"])),
                                MemoryControlConfig(**checkpoint["config"]))
    agent.load_state_dict(checkpoint["model"], strict=True)
    agent.encoder_identity = encoder_hash(agent.encoder)
    if agent.encoder_identity != checkpoint["encoder_hash"]:
        raise ValueError("Checkpoint visual encoder identity does not match weights")
    return agent.to(device).eval()
