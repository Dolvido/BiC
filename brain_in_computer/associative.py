"""Learned visual matching with bounded, persistent episodic associations.

This is a hybrid engineering model inspired by rapid hippocampal association.
Labels are explicit symbolic addresses supplied by the teacher; arbitrary names
are not learned through language-network weights. Only pixels enter the learned
visual matcher. Teaching stores exemplars without changing the encoder, which
protects earlier representations but is not neural consolidation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
from torch.nn import functional as F

SCHEMA = "bic-associative-memory-v1"
ENCODER_SCHEMA = "bic-glyph-encoder-v1"


@dataclass(frozen=True)
class EncoderConfig:
    embedding_size: int = 64


class GlyphEncoder(nn.Module):
    """Small visual association network; inputs are RGB object crops, not IDs."""

    def __init__(self, config: EncoderConfig | None = None):
        super().__init__()
        self.config = config or EncoderConfig()
        if type(self.config.embedding_size) is not int or not 8 <= self.config.embedding_size <= 1024:
            raise ValueError("embedding_size must be between 8 and 1024")
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
        )
        self.projection = nn.Linear(128, self.config.embedding_size)
        # Auxiliary reconstruction target is training-only; inference uses pixels.
        self.occupancy_head = nn.Linear(128, 16)

    def encode(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        if pixels.ndim != 4 or pixels.shape[1:] != (3, 32, 32):
            raise ValueError("pixels must have shape [batch,3,32,32]")
        features = self.features(pixels)
        return F.normalize(self.projection(features), dim=-1), self.occupancy_head(features)

    def forward(self, pixels: Tensor) -> Tensor:
        return self.encode(pixels)[0]


def distinct_patterns(count: int, seed: int, exclude: Iterable[int] = ()) -> list[int]:
    """Teacher-side random 4x4 occupancy identities; never an inference input."""
    if type(count) is not int or count < 0:
        raise ValueError("count must be nonnegative")
    blocked = set(exclude)
    possible = [n for n in range(1, 65535) if 5 <= n.bit_count() <= 11 and n not in blocked]
    if count > len(possible):
        raise ValueError("requested more distinct patterns than exist")
    return random.Random(seed).sample(possible, count)


def render_glyph(pattern: int, seed: int = 0, variation: bool = True) -> Tensor:
    """Rasterize an object for the controlled curriculum, returning RGB [3,32,32]."""
    if type(pattern) is not int or not 0 < pattern < 65536:
        raise ValueError("pattern must be a nonempty 16-bit occupancy mask")
    rng = random.Random(seed)
    bg = tuple(rng.randint(4, 28) for _ in range(3)) if variation else (12, 12, 18)
    fg = tuple(rng.randint(110, 250) for _ in range(3)) if variation else (225, 230, 240)
    cell = rng.choice((4, 5, 6)) if variation else 5
    dx, dy = (rng.randint(-2, 2), rng.randint(-2, 2)) if variation else (0, 0)
    left, top = (32 - cell * 4) // 2 + dx, (32 - cell * 4) // 2 + dy
    canvas = Image.new("RGB", (32, 32), bg)
    draw = ImageDraw.Draw(canvas)
    for index in range(16):
        if pattern & (1 << index):
            x, y = left + index % 4 * cell, top + index // 4 * cell
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=fg)
    return torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float().div_(255)


def encoder_hash(encoder: GlyphEncoder) -> str:
    """Bind a memory to exact architecture and encoder weights, across devices."""
    digest = hashlib.sha256(json.dumps(asdict(encoder.config), sort_keys=True).encode())
    for name, value in sorted(encoder.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_encoder_checkpoint(path: str | Path, device: str = "cpu") -> GlyphEncoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != ENCODER_SCHEMA:
        raise ValueError("unsupported visual encoder checkpoint schema")
    encoder = GlyphEncoder(EncoderConfig(**checkpoint["config"]))
    encoder.load_state_dict(checkpoint["model"], strict=True)
    if encoder_hash(encoder) != checkpoint["encoder_hash"]:
        raise ValueError("encoder checkpoint hash does not match its weights")
    encoder.to(device).eval()
    # Reading new names never performs optimization.
    encoder.requires_grad_(False)
    return encoder


def load_calibration(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != ENCODER_SCHEMA:
        raise ValueError("unsupported visual encoder checkpoint schema")
    return dict(checkpoint["calibration"])


class AssociativeMemory:
    """Bounded label->visual-exemplar bank with ambiguity and unknown rejection.

    ``confidence`` is cosine similarity, NOT a calibrated probability. Threshold
    and margin are validated only for this synthetic crop distribution. Capacity
    is in distinct labels. A full bank rejects new labels until explicit forget.
    """

    def __init__(self, encoder: GlyphEncoder, threshold: float = .85,
                 margin: float = .03, capacity: int = 128, examples_per_label: int = 4):
        if not isinstance(encoder, GlyphEncoder):
            raise TypeError("encoder must be a GlyphEncoder")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be a finite cosine similarity in [-1,1]")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or not 0 <= margin <= 2:
            raise ValueError("margin must be in [0,2]")
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("capacity must be in [1,4096]")
        if type(examples_per_label) is not int or not 1 <= examples_per_label <= 32:
            raise ValueError("examples_per_label must be in [1,32]")
        self.encoder = encoder
        self.threshold, self.margin = float(threshold), float(margin)
        self.capacity, self.examples_per_label = capacity, examples_per_label
        self.model_hash = encoder_hash(encoder)
        self._entries: dict[str, list[Tensor]] = {}

    @property
    def labels(self) -> list[str]:
        return list(self._entries)

    @staticmethod
    def _label(label: str) -> str:
        if not isinstance(label, str) or not label.strip() or len(label) > 80 or any(ord(c) < 32 for c in label):
            raise ValueError("label must be 1-80 visible characters")
        return label.strip()

    @torch.no_grad()
    def _embed(self, pixels: Tensor) -> Tensor:
        if encoder_hash(self.encoder) != self.model_hash:
            raise ValueError("encoder changed: old memory embeddings require the original weights")
        if not isinstance(pixels, Tensor):
            raise TypeError("pixels must be a torch Tensor")
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.ndim != 4 or pixels.shape[1:] != (3, 32, 32) or len(pixels) < 1:
            raise ValueError("pixels must be [3,32,32] or [N,3,32,32]")
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("pixels must be finite floating-point RGB in [0,1]")
        self.encoder.eval()
        return self.encoder(pixels.to(next(self.encoder.parameters()).device)).detach().cpu()

    def teach(self, label: str, pixels: Tensor) -> dict:
        label = self._label(label)
        if label not in self._entries and len(self._entries) >= self.capacity:
            raise ValueError("memory capacity reached; explicitly forget a label before adding one")
        if not isinstance(pixels, Tensor) or pixels.shape != (3, 32, 32):
            raise ValueError("teach requires one object crop [3,32,32]")
        embedding = self._embed(pixels)[0]
        exemplars = self._entries.get(label, []) + [embedding]
        self._entries[label] = exemplars[-self.examples_per_label:]
        return {"label": label, "examples": len(self._entries[label]), "labels": len(self._entries)}

    def forget(self, label: str) -> bool:
        label = self._label(label)
        return self._entries.pop(label, None) is not None

    def evidence(self, label: str, candidate_pixels: Tensor | list[Tensor]) -> dict:
        """Return all cosine similarities, with no threshold or action selection.

        This is the persistent input for regional control. Symbolic lookup only
        addresses the taught exemplars; the downstream networks receive every
        candidate's evidence and choose what to do themselves.
        """
        label = self._label(label)
        if isinstance(candidate_pixels, list):
            if not candidate_pixels:
                raise ValueError("at least one candidate is required")
            candidate_pixels = torch.stack(candidate_pixels)
        embeddings = self._embed(candidate_pixels)
        if label not in self._entries:
            return {"scores": torch.zeros(len(embeddings)), "known": False}
        scores = (embeddings @ torch.stack(self._entries[label]).T).max(dim=1).values
        return {"scores": scores.clamp(-1, 1), "known": True}

    def _decision(self, scores: Tensor) -> dict:
        order = scores.argsort(descending=True)
        best = float(scores[order[0]])
        gap = best - float(scores[order[1]]) if len(scores) > 1 else 2.0
        # Exact ties always abstain, including a user-selected zero margin.
        second_matches = len(scores) > 1 and float(scores[order[1]]) >= self.threshold
        accepted = best >= self.threshold and not second_matches and gap >= self.margin and gap > 1e-7
        return {"index": int(order[0]) if accepted else None,
                "confidence": best, "margin": gap,
                "status": "matched" if accepted else ("unknown" if best < self.threshold else "ambiguous")}

    def find(self, label: str, candidate_pixels: Tensor | list[Tensor]) -> dict:
        label = self._label(label)
        if label not in self._entries:
            return {"index": None, "confidence": None, "margin": None, "status": "unknown_label"}
        if isinstance(candidate_pixels, list):
            if not candidate_pixels:
                raise ValueError("at least one candidate is required")
            candidate_pixels = torch.stack(candidate_pixels)
        embeddings = self._embed(candidate_pixels)
        scores = (embeddings @ torch.stack(self._entries[label]).T).max(dim=1).values
        return self._decision(scores)

    def name(self, pixels: Tensor) -> dict:
        if not self._entries:
            return {"label": None, "confidence": None, "margin": None, "status": "empty_memory"}
        if not isinstance(pixels, Tensor) or pixels.shape != (3, 32, 32):
            raise ValueError("name requires one object crop [3,32,32]")
        query = self._embed(pixels)[0]
        labels = self.labels
        scores = torch.stack([(torch.stack(self._entries[label]) @ query).max() for label in labels])
        result = self._decision(scores)
        index = result.pop("index")
        result["label"] = labels[index] if index is not None else None
        return result

    def save(self, path: str | Path) -> None:
        if encoder_hash(self.encoder) != self.model_hash:
            raise ValueError("encoder changed: refusing to save stale memory")
        content = {"schema": SCHEMA, "encoder_hash": self.model_hash,
                   "embedding_size": self.encoder.config.embedding_size,
                   "threshold": self.threshold, "margin": self.margin,
                   "capacity": self.capacity, "examples_per_label": self.examples_per_label,
                   "entries": {label: [v.tolist() for v in values] for label, values in self._entries.items()}}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, encoder: GlyphEncoder) -> "AssociativeMemory":
        path = Path(path)
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("memory file is larger than the supported bounded format")
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict) or content.get("schema") != SCHEMA:
            raise ValueError("unsupported associative memory schema")
        if content.get("encoder_hash") != encoder_hash(encoder):
            raise ValueError("memory belongs to different encoder weights")
        if content.get("embedding_size") != encoder.config.embedding_size:
            raise ValueError("memory embedding shape does not match encoder")
        memory = cls(encoder, content["threshold"], content["margin"], content["capacity"], content["examples_per_label"])
        entries = content.get("entries")
        if not isinstance(entries, dict) or len(entries) > memory.capacity:
            raise ValueError("invalid memory label count")
        for label, values in entries.items():
            if cls._label(label) != label or not isinstance(values, list) or not 1 <= len(values) <= memory.examples_per_label:
                raise ValueError("invalid memory label or exemplar count")
            vectors = torch.tensor(values, dtype=torch.float32)
            if vectors.shape != (len(values), encoder.config.embedding_size) or not torch.isfinite(vectors).all():
                raise ValueError("invalid memory embedding")
            if not torch.allclose(vectors.norm(dim=-1), torch.ones(len(values)), atol=1e-4, rtol=1e-4):
                raise ValueError("memory embedding must be normalized")
            memory._entries[label] = list(vectors.unbind())
        return memory
