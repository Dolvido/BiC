"""Shared neural recall learned by replaying rapidly acquired visual memories.

The known-name set is an explicit vocabulary gate. Visual associations live in
shared byte/GRU/MLP weights, not in a label-indexed vector table. Consolidation is
supervised rehearsal into a separate network; it does not model biological sleep.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .associative import AssociativeMemory, GlyphEncoder, encoder_hash
from .regional_memory import RegionalMemoryAgent, normalize_instruction

SCHEMA = "bic-consolidated-recall-v1"


@dataclass(frozen=True)
class RecallConfig:
    byte_features: int = 24
    hidden_size: int = 96
    visual_features: int = 64

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or not 8 <= value <= 1024:
                raise ValueError(f"{key} must be an integer from 8 through 1024")


class NeuralRecall(nn.Module):
    """Encode UTF-8 bytes with parameters shared by every taught object name."""

    def __init__(self, config: RecallConfig | None = None):
        super().__init__()
        self.config = config or RecallConfig()
        self.bytes = nn.Embedding(257, self.config.byte_features, padding_idx=0)
        self.sequence = nn.GRU(self.config.byte_features, self.config.hidden_size,
                               batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Tanh(),
            nn.Linear(self.config.hidden_size, self.config.visual_features),
        )

    def forward(self, names: Sequence[str]) -> Tensor:
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence) or not names:
            raise ValueError("names must be a nonempty sequence of object names")
        encoded = [AssociativeMemory._label(name).encode("utf-8") for name in names]
        device = next(self.parameters()).device
        tokens = torch.zeros(len(names), max(map(len, encoded)), device=device, dtype=torch.long)
        for index, raw in enumerate(encoded):
            tokens[index, :len(raw)] = torch.tensor([byte + 1 for byte in raw], device=device)
        output, _ = self.sequence(self.bytes(tokens))
        lengths = torch.tensor([len(raw) - 1 for raw in encoded], device=device)
        final = output[torch.arange(len(names), device=device), lengths]
        return F.normalize(self.projection(final), dim=-1)


def replay_examples(memory: AssociativeMemory, labels: Sequence[str] | None = None):
    """Training-only export of episodic targets; never called during recall."""
    if not isinstance(memory, AssociativeMemory):
        raise TypeError("memory must be an AssociativeMemory")
    if encoder_hash(memory.encoder) != memory.model_hash:
        raise ValueError("episodic memory encoder has changed")
    labels = memory.labels if labels is None else list(labels)
    if not labels or any(label not in memory.labels for label in labels):
        raise ValueError("replay requires taught labels")
    names, targets = [], []
    for label in labels:
        for exemplar in memory._entries[label]:
            names.append(label)
            targets.append(exemplar.clone())
    return names, torch.stack(targets)


class ConsolidatedMemory:
    """Inference owns shared learned weights, a vocabulary set and visual encoder.

    It owns no original memory bank, per-name visual vectors, object identities,
    teacher labels or exemplar images. Unknown rejection uses the explicit set;
    known-name association is computed by NeuralRecall's optimizer-trained weights.
    """

    def __init__(self, recall: NeuralRecall, encoder: GlyphEncoder, labels: Sequence[str]):
        if not isinstance(recall, NeuralRecall) or not isinstance(encoder, GlyphEncoder):
            raise TypeError("recall and encoder must be NeuralRecall and GlyphEncoder")
        if recall.config.visual_features != encoder.config.embedding_size:
            raise ValueError("recall and encoder embedding dimensions must agree")
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
            raise ValueError("labels must be a sequence")
        normalized = [AssociativeMemory._label(label) for label in labels]
        if len(set(normalized)) != len(normalized) or len(normalized) > 4096:
            raise ValueError("known labels must be unique and number at most 4096")
        if next(recall.parameters()).dtype != torch.float32 or next(encoder.parameters()).dtype != torch.float32:
            raise ValueError("consolidated recall requires float32 networks")
        self.recall = recall.eval()
        self.encoder = encoder.eval()
        self.labels = tuple(normalized)
        self._known = frozenset(normalized)
        self.encoder_identity = encoder_hash(encoder)

    @torch.no_grad()
    def evidence(self, label: str, candidate_pixels: Tensor | list[Tensor]) -> dict:
        label = AssociativeMemory._label(label)
        if encoder_hash(self.encoder) != self.encoder_identity:
            raise ValueError("consolidated recall requires its original frozen visual encoder")
        if isinstance(candidate_pixels, list):
            if not candidate_pixels:
                raise ValueError("at least one candidate is required")
            candidate_pixels = torch.stack(candidate_pixels)
        if not isinstance(candidate_pixels, Tensor) or candidate_pixels.ndim != 4 or candidate_pixels.shape[1:] != (3, 32, 32) or len(candidate_pixels) < 1:
            raise ValueError("candidates must have shape [N,3,32,32]")
        if not candidate_pixels.is_floating_point() or not torch.isfinite(candidate_pixels).all() or candidate_pixels.min() < 0 or candidate_pixels.max() > 1:
            raise ValueError("candidate pixels must be finite floating-point RGB in [0,1]")
        if label not in self._known:
            return {"scores": torch.zeros(len(candidate_pixels)), "known": False}
        self.encoder.eval()
        self.recall.eval()
        visual_parameter = next(self.encoder.parameters())
        embeddings = self.encoder(candidate_pixels.to(visual_parameter.device, visual_parameter.dtype))
        prediction = self.recall([label]).to(embeddings.device)
        scores = (embeddings @ prediction.T).squeeze(1).clamp(-1, 1)
        return {"scores": scores.cpu(), "known": True}

    @torch.no_grad()
    def act(self, agent: RegionalMemoryAgent, instruction: str,
            candidate_pixels: Tensor | list[Tensor]) -> dict:
        if not isinstance(agent, RegionalMemoryAgent) or agent.encoder_identity != self.encoder_identity:
            raise ValueError("regional agent must use the same frozen visual encoder")
        label, normalized = normalize_instruction(instruction)
        evidence = self.evidence(label, candidate_pixels)
        if evidence["scores"].shape != (4,):
            raise ValueError("regional control requires four crops in row-major order")
        parameter = next(agent.parameters())
        scores = evidence["scores"].unsqueeze(0).to(parameter.device, parameter.dtype)
        known = scores.new_tensor([[float(evidence["known"])]])
        modes = [(module, module.training) for module in agent.modules()]
        agent.eval()
        try:
            output = agent.forward_evidence(scores, known, [normalized])
            action = int(output["logits"][0, -1].argmax())
            return {"action": action, "index": action if action < 4 else None,
                    "reply": agent._decode(output)[0], "known_label": evidence["known"],
                    "scores": scores[0].tolist(), "action_source": "regional_motor_cortex",
                    "response_source": "neural_byte_decoder", "memory_source": "shared_neural_weights"}
        finally:
            for module, mode in modes:
                module.training = mode

    def save(self, path: str | Path, metadata: dict | None = None):
        if encoder_hash(self.encoder) != self.encoder_identity:
            raise ValueError("visual encoder changed")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA, "config": asdict(self.recall.config),
                   "encoder_hash": self.encoder_identity, "known_names": list(self.labels),
                   "model": {key: value.detach().cpu() for key, value in self.recall.state_dict().items()},
                   "metadata": metadata or {}}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, encoder: GlyphEncoder, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("unsupported consolidated recall checkpoint")
        if payload["encoder_hash"] != encoder_hash(encoder):
            raise ValueError("consolidated checkpoint uses a different visual encoder")
        recall = NeuralRecall(RecallConfig(**payload["config"]))
        recall.load_state_dict(payload["model"], strict=True)
        if not all(torch.isfinite(value).all() for value in recall.state_dict().values()):
            raise ValueError("nonfinite recall weights")
        return cls(recall.to(device).eval(), encoder, payload["known_names"])
