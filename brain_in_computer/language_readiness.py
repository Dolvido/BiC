"""Inference-only inspection of randomly initialized language interfaces.

This module has no optimizer, training loop, data loader, or model downloader.
The optional checkpoint supplies only previously trained symbolic-core weights.
"""

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from .language import ByteCodec, LanguageBrain, LanguageConfig
from .model import Brain, BrainConfig
from .resources import estimate_language_budget
from .tasks import TaskStream


def readiness_report(checkpoint=None, config=None, device="cpu", seed=17):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("readiness device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    checkpoint_hash = None
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if checkpoint:
            path = Path(checkpoint)
            checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("schema_version") != 1:
                raise ValueError("Unsupported symbolic-core checkpoint schema")
            brain = Brain(BrainConfig(**saved["brain_config"]))
            brain.load_state_dict(saved["model"], strict=True)
        else:
            brain = Brain()
        model = LanguageBrain(brain, config or LanguageConfig()).to(device).eval()
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        # Both rows have identical sensory observations and no legacy instruction.
        # Only the utterance differs; this probes a causal interface, not meaning.
        raw = TaskStream(123).sample(1, delay=0, device=device).observations
        observations = {key: value.repeat(2, *([1]*(value.ndim-1))) for key, value in raw.items()}
        observations["tokens"].zero_()
        ids, lengths = model.prepare_inputs(["Look at the red cube.", "Look at the blue cube."])
        changed_ids, changed_lengths = model.prepare_inputs(["Find the green object.", "Ignore the picture."])
        decoder = torch.full((2, 1), ByteCodec.BOS, dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(observations, ids, lengths, decoder)
            changed = model(observations, changed_ids, changed_lengths, decoder)
            lesioned = model(observations, ids, lengths, decoder, ablate=("temporal_language",))
            changed_lesioned = model(observations, changed_ids, changed_lengths, decoder,
                                     ablate=("temporal_language",))
        unchanged = all(torch.equal(before[k], value) for k, value in model.state_dict().items())
        counts = model.parameter_counts()
        count = sum(p.numel() for p in model.parameters())
        results = {
            "version": "0.2.0", "device": str(device), "seed": seed,
            "status": "Untrained language interfaces; English competence is not established",
            "language_training_performed": False, "english_optimizer_steps": 0,
            "model_downloaded": False, "language_checkpoint_saved": False,
            "symbolic_core_checkpoint_sha256": checkpoint_hash,
            "language_config": asdict(model.config), "parameter_counts": counts,
            "total_parameters": count,
            "output_shapes": {k: list(output[k].shape) for k in ("logits", "language_logits", "concept_context")},
            "checks": {"all_output_logits_finite": bool(torch.isfinite(output["language_logits"]).all()),
                       "weights_unchanged": unchanged,
                       "utterance_changes_core_logits": not torch.equal(output["logits"], changed["logits"]),
                       "utterance_changes_production_logits": not torch.equal(output["language_logits"], changed["language_logits"]),
                       "temporal_lesion_blocks_text_to_production": torch.equal(lesioned["language_logits"], changed_lesioned["language_logits"])},
            # Compare the same batch positions across interventions. Equal
            # inputs in different rows can differ by float32 BLAS roundoff on
            # some CPUs; that is not a causal influence of the utterance.
            "comparison_method": "same batch positions, changed utterances, identical observations",
            "lesion_max_absolute_change": float((lesioned["language_logits"]-changed_lesioned["language_logits"]).abs().max()),
            "parameter_budget": estimate_language_budget(count),
            "limitations": ["Input differences affect untrained random networks; this is not evidence of English comprehension.",
                            "The complete utterance must be available before the sensory/action episode starts.",
                            "Production is conditioned on final core state; no free-running text generation or language trainer is exposed.",
                            "Anatomical names describe functional inspirations, not one-to-one maps of human language regions."]}
    if not all(results["checks"].values()):
        raise RuntimeError(f"Language readiness checks failed: {results['checks']}")
    return results
