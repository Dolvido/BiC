"""A compact modular brain-inspired learning system, not a demonstrated AGI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .regions import (
    Amygdala, BasalGanglia, Cerebellum, Hippocampus, Hypothalamus,
    MotorCortex, ParietalAssociation, PrefrontalCortex, SensoryCortex,
    SomatosensoryCortex, TemporalLanguage, Thalamus,
)


REGION_NAMES = (
    "visual_cortex", "auditory_cortex", "somatosensory_cortex",
    "temporal_language", "parietal_association", "thalamus",
    "prefrontal_cortex", "hippocampus", "basal_ganglia", "motor_cortex",
    "cerebellum", "amygdala", "hypothalamus",
)


# Each triple is (source, destination, delay_in_observation_steps).
# Multiple inputs to a destination are combined by that destination's own network.
# External sensory inputs and within-region recurrence are described separately.
CONNECTOME = (
    ("prefrontal_cortex", "visual_cortex", 1),
    ("prefrontal_cortex", "auditory_cortex", 1),
    ("prefrontal_cortex", "temporal_language", 1),
    ("motor_cortex", "somatosensory_cortex", 1),
    ("visual_cortex", "parietal_association", 0),
    ("auditory_cortex", "parietal_association", 0),
    ("somatosensory_cortex", "parietal_association", 0),
    ("temporal_language", "parietal_association", 0),
    ("visual_cortex", "amygdala", 0),
    ("auditory_cortex", "amygdala", 0),
    ("somatosensory_cortex", "amygdala", 0),
    ("somatosensory_cortex", "hypothalamus", 0),
    ("visual_cortex", "thalamus", 0),
    ("auditory_cortex", "thalamus", 0),
    ("somatosensory_cortex", "thalamus", 0),
    ("temporal_language", "thalamus", 0),
    ("parietal_association", "thalamus", 0),
    ("prefrontal_cortex", "thalamus", 1),
    ("amygdala", "thalamus", 0),
    ("hypothalamus", "thalamus", 0),
    ("parietal_association", "hippocampus", 0),
    ("temporal_language", "hippocampus", 0),
    ("somatosensory_cortex", "hippocampus", 0),
    ("prefrontal_cortex", "hippocampus", 1),
    ("amygdala", "hippocampus", 0),
    ("thalamus", "prefrontal_cortex", 0),
    ("hippocampus", "prefrontal_cortex", 0),
    ("temporal_language", "prefrontal_cortex", 0),
    ("cerebellum", "prefrontal_cortex", 1),
    ("amygdala", "prefrontal_cortex", 0),
    ("hypothalamus", "prefrontal_cortex", 0),
    ("prefrontal_cortex", "basal_ganglia", 0),
    ("hippocampus", "basal_ganglia", 0),
    ("amygdala", "basal_ganglia", 0),
    ("prefrontal_cortex", "motor_cortex", 0),
    ("thalamus", "motor_cortex", 0),
    ("basal_ganglia", "motor_cortex", 0),
    ("cerebellum", "motor_cortex", 1),
    ("prefrontal_cortex", "cerebellum", 0),
    ("motor_cortex", "cerebellum", 0),
    ("somatosensory_cortex", "cerebellum", 0),
)


# Each triple is (observation_channel, destination, role). The cerebellar
# prediction error is computed from the current raw channel and the previous
# prediction, so it remains available when a sensory cortex is silenced.
EXTERNAL_INPUTS = (
    ("visual", "visual_cortex", "sensory_encoding"),
    ("visual", "cerebellum", "prediction_error"),
    ("auditory", "auditory_cortex", "sensory_encoding"),
    ("auditory", "cerebellum", "prediction_error"),
    ("body", "somatosensory_cortex", "body_encoding"),
    ("body", "cerebellum", "prediction_error"),
    ("feedback", "somatosensory_cortex", "external_feedback"),
    ("tokens", "temporal_language", "instruction_embedding"),
    ("language_context", "temporal_language", "optional_preencoded_utterance"),
    ("memory_context", "hippocampus", "optional_persistent_recall_evidence"),
)


@dataclass(frozen=True)
class BrainConfig:
    hidden_size: int = 48
    visual_dim: int = 8
    auditory_dim: int = 8
    body_dim: int = 4
    vocab_size: int = 8
    num_actions: int = 4
    memory_slots: int = 8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")

    @property
    def observation_dim(self) -> int:
        return self.visual_dim + self.auditory_dim + self.body_dim


@dataclass(frozen=True)
class BrainState:
    """Caller-owned recurrent state; never stored on the model itself.

    Tensor rows belong to independent sessions. Keep their ordering stable across
    calls. Graphs span calls by default; use ``detached`` at a deliberate training
    boundary. A detached state shares tensor storage, as Tensor.detach does.
    """

    previous_prefrontal: Tensor
    previous_motor: Tensor
    previous_cerebellum: Tensor
    previous_prediction: Tensor
    memory: tuple[Tensor, ...] = ()

    def detached(self) -> BrainState:
        return BrainState(
            self.previous_prefrontal.detach(), self.previous_motor.detach(),
            self.previous_cerebellum.detach(), self.previous_prediction.detach(),
            tuple(value.detach() for value in self.memory),
        )


class Brain(nn.Module):
    """Thirteen regional networks connected by explicit tensor pathways.

    Inputs are batched sequences. ``forward`` resets state for each call;
    ``forward_with_state`` continues an explicitly supplied caller-owned state.
    All communication consists of hidden vectors on the explicit directed graph
    above. Gradients include bounded hippocampal writes and may span chunks.

    ``ablate`` silences all outgoing tensors of the named regions throughout the
    whole episode, including task heads. It does not retrain or compensate.
    """

    connectome = CONNECTOME
    external_inputs = EXTERNAL_INPUTS
    region_names = REGION_NAMES

    def __init__(self, config: BrainConfig | None = None):
        super().__init__()
        self.config = config or BrainConfig()
        c, h = self.config, self.config.hidden_size
        self.regions = nn.ModuleDict({
            "visual_cortex": SensoryCortex(c.visual_dim, h, c.num_actions),
            "auditory_cortex": SensoryCortex(c.auditory_dim, h, c.num_actions),
            "somatosensory_cortex": SomatosensoryCortex(c.body_dim, h),
            "temporal_language": TemporalLanguage(c.vocab_size, h),
            "parietal_association": ParietalAssociation(h),
            "thalamus": Thalamus(h),
            "prefrontal_cortex": PrefrontalCortex(h),
            "hippocampus": Hippocampus(h, c.memory_slots),
            "basal_ganglia": BasalGanglia(h, c.num_actions),
            "motor_cortex": MotorCortex(h, c.num_actions),
            "cerebellum": Cerebellum(c.observation_dim, h),
            "amygdala": Amygdala(h),
            "hypothalamus": Hypothalamus(h),
        })

    def region_parameter_counts(self) -> dict[str, int]:
        return {name: sum(p.numel() for p in region.parameters()) for name, region in self.regions.items()}

    def initial_state(self, batch_size: int) -> BrainState:
        """Create zero state using this model's current device and dtype."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        parameter = next(self.parameters())
        hidden = (batch_size, self.config.hidden_size)
        return BrainState(
            parameter.new_zeros(hidden), parameter.new_zeros(hidden),
            parameter.new_zeros(hidden),
            parameter.new_zeros(batch_size, self.config.observation_dim), (),
        )

    def _validate_state(self, state: BrainState, batch_size: int, *, match_model=True) -> None:
        if not isinstance(state, BrainState):
            raise ValueError("state must be a BrainState or None")
        if not isinstance(state.memory, tuple) or len(state.memory) > self.config.memory_slots:
            raise ValueError("state memory must be a tuple within the configured memory_slots")
        reference = next(self.parameters()) if match_model else state.previous_prefrontal
        if not isinstance(reference, Tensor) or not reference.is_floating_point() or reference.device.type == "meta":
            raise ValueError("state must contain materialized floating-point tensors")
        fields = [("previous_prefrontal", state.previous_prefrontal, self.config.hidden_size),
                  ("previous_motor", state.previous_motor, self.config.hidden_size),
                  ("previous_cerebellum", state.previous_cerebellum, self.config.hidden_size),
                  ("previous_prediction", state.previous_prediction, self.config.observation_dim)]
        fields.extend((f"memory[{index}]", value, self.config.hidden_size)
                      for index, value in enumerate(state.memory))
        for name, value, width in fields:
            if not isinstance(value, Tensor) or value.shape != (batch_size, width):
                raise ValueError(f"state {name} must have shape {(batch_size, width)}")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"state {name} must match the {'model' if match_model else 'state'} device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"state {name} must contain only finite values")

    def state_to_dict(self, state: BrainState) -> dict[str, Tensor]:
        """Export independent CPU tensors suitable for torch.load(weights_only=True).

        This records activity only, not model weights, observations or a session
        identity. Save the matching model checkpoint/config separately. Export
        detaches gradients; it is a persistence boundary, not training resume.
        """
        if not isinstance(state, BrainState) or not isinstance(state.previous_prefrontal, Tensor) or state.previous_prefrontal.ndim != 2:
            raise ValueError("state must have a two-dimensional prefrontal tensor")
        batch = state.previous_prefrontal.shape[0]
        if batch < 1:
            raise ValueError("state batch must be nonempty")
        self._validate_state(state, batch)
        memory = (torch.stack(state.memory, dim=1) if state.memory else
                  state.previous_prefrontal.new_empty(batch, 0, self.config.hidden_size))
        values = {name: getattr(state, name) for name in (
            "previous_prefrontal", "previous_motor", "previous_cerebellum", "previous_prediction")}
        values["memory"] = memory
        return {"schema_version": torch.tensor(1, dtype=torch.int64, device="cpu"),
                **{name: value.detach().cpu().clone() for name, value in values.items()}}

    def state_from_dict(self, payload: Mapping[str, Tensor]) -> BrainState:
        """Validate saved activity and copy it to this model's device and dtype.

        Only the tensor-only version-1 format is accepted. Shape compatibility
        does not establish that activity came from the same learned weights;
        checkpoint provenance remains the caller's responsibility.
        """
        names = ("previous_prefrontal", "previous_motor", "previous_cerebellum", "previous_prediction")
        if not isinstance(payload, Mapping) or set(payload) != {*names, "memory", "schema_version"}:
            raise ValueError("Invalid brain state tensor dictionary fields")
        version = payload["schema_version"]
        if (not isinstance(version, Tensor) or version.shape != () or version.dtype != torch.int64
                or version.device.type == "meta" or version.item() != 1):
            raise ValueError("Unsupported brain state schema_version")
        first, memory = payload["previous_prefrontal"], payload["memory"]
        if not isinstance(first, Tensor) or first.ndim != 2 or first.shape[0] < 1:
            raise ValueError("Saved state must have a nonempty batch")
        if first.device.type == "meta" or not first.is_floating_point():
            raise ValueError("Saved state must contain materialized floating-point tensors")
        batch = first.shape[0]
        if (not isinstance(memory, Tensor) or memory.ndim != 3 or memory.shape[0] != batch
                or memory.shape[2] != self.config.hidden_size or memory.shape[1] > self.config.memory_slots):
            raise ValueError("Saved memory must have shape [batch, slots <= memory_slots, hidden_size]")
        if memory.device != first.device or memory.dtype != first.dtype or not torch.isfinite(memory).all():
            raise ValueError("Saved memory must be finite and match state device and dtype")
        restored = BrainState(*(payload[name] for name in names), tuple(memory.unbind(dim=1)))
        self._validate_state(restored, batch, match_model=False)
        parameter = next(self.parameters())
        def copy(value):
            return value.detach().to(device=parameter.device, dtype=parameter.dtype).clone()
        result = BrainState(*(copy(getattr(restored, name)) for name in names),
                            tuple(copy(value) for value in restored.memory))
        self._validate_state(result, batch)
        return result

    def _validate_observations(self, observations: Mapping[str, Tensor]) -> tuple[int, int]:
        expected = {
            "visual": self.config.visual_dim, "auditory": self.config.auditory_dim,
            "body": self.config.body_dim, "feedback": 2,
        }
        missing = set(expected).union({"tokens"}) - observations.keys()
        if missing:
            raise ValueError(f"Missing observation channels: {sorted(missing)}")
        visual = observations["visual"]
        if not isinstance(visual, Tensor) or visual.ndim != 3:
            raise ValueError("visual must have shape [batch, time, visual_dim]")
        batch, length = visual.shape[:2]
        if batch == 0 or length == 0:
            raise ValueError("Observation batch and time dimensions must be nonempty")
        parameter = next(self.parameters())
        for name, width in expected.items():
            tensor = observations[name]
            if not isinstance(tensor, Tensor) or tensor.shape != (batch, length, width):
                raise ValueError(f"{name} must have shape {(batch, length, width)}")
            if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
                raise ValueError(f"{name} must use model device {parameter.device} and dtype {parameter.dtype}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
        tokens = observations["tokens"]
        if not isinstance(tokens, Tensor) or tokens.shape != (batch, length) or tokens.dtype != torch.long:
            raise ValueError("tokens must be a torch.long tensor of shape [batch, time]")
        if tokens.device != parameter.device:
            raise ValueError(f"tokens must use model device {parameter.device}")
        if not ((tokens >= 0) & (tokens < self.config.vocab_size)).all():
            raise ValueError(f"tokens must be in the range [0, {self.config.vocab_size})")
        return batch, length

    def forward(
        self, observations: Mapping[str, Tensor], ablate: Iterable[str] = (),
        *, return_activity: bool = False, language_context: Tensor | None = None,
        memory_context: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Evaluate independent episodes with fresh state; preserve legacy outputs."""
        result, _ = self.forward_with_state(
            observations, None, ablate, return_activity=return_activity,
            language_context=language_context, memory_context=memory_context,
        )
        return result

    def forward_with_state(
        self, observations: Mapping[str, Tensor], state: BrainState | None = None,
        ablate: Iterable[str] = (), *, return_activity: bool = False,
        language_context: Tensor | None = None, memory_context: Tensor | None = None,
    ) -> tuple[dict[str, Tensor | dict[str, Tensor]], BrainState]:
        """Continue explicit state without mutating it or storing session activity.

        ``None`` starts independent episodes. State remains connected to its
        graph unless the caller detaches it. Splitting a sequence into chunks
        with the same ablations/contexts preserves its original computation.

        A sensory-region ablation is not removal of that input modality: raw
        observations also reach the cerebellum through its prediction-error
        channel. See ``external_inputs`` for all direct observation pathways.

        Optional ``language_context`` is [batch, time, hidden_size]. Its producer
        must only encode information already available at the corresponding
        observation. Omitting it preserves the original symbolic computation.

        Optional ``memory_context`` has the same shape and enters the learned
        hippocampal output transformation. Its producer must use only stored
        evidence and current/past observations. A hippocampal lesion silences
        this input together with within-episode memory; None preserves legacy
        behavior and adds no parameters to existing checkpoint schemas.
        """
        if isinstance(ablate, str):
            raise ValueError("ablate must be an iterable of region names, not a single string")
        disabled = frozenset(ablate)
        unknown = disabled - set(REGION_NAMES)
        if unknown:
            raise ValueError(f"Unknown regions for ablation: {sorted(unknown)}")
        batch, length = self._validate_observations(observations)
        for context_name, context in (("language_context", language_context),
                                      ("memory_context", memory_context)):
            if context is None:
                continue
            expected_shape = (batch, length, self.config.hidden_size)
            reference = observations["visual"]
            if not isinstance(context, Tensor) or context.shape != expected_shape:
                raise ValueError(f"{context_name} must have shape {expected_shape}")
            if context.device != reference.device or context.dtype != reference.dtype:
                raise ValueError(f"{context_name} must match the model device and dtype")
            if not torch.isfinite(context).all():
                raise ValueError(f"{context_name} must contain only finite values")
        zero = observations["visual"].new_zeros(batch, self.config.hidden_size)
        state = self.initial_state(batch) if state is None else state
        self._validate_state(state, batch)
        # A newly requested lesion also silences activity carried from an earlier
        # intact call. Never overwrite the caller's tensors while doing so.
        previous_prefrontal = zero if "prefrontal_cortex" in disabled else state.previous_prefrontal
        previous_motor = zero if "motor_cortex" in disabled else state.previous_motor
        previous_cerebellum = zero if "cerebellum" in disabled else state.previous_cerebellum
        previous_prediction = (zero.new_zeros(batch, self.config.observation_dim)
                               if "cerebellum" in disabled else state.previous_prediction)
        memory = () if "hippocampus" in disabled else state.memory
        outputs: dict[str, list[Tensor]] = {
            key: [] for key in ("logits", "visual_logits", "auditory_logits", "prediction", "value")
        }
        activities: dict[str, list[Tensor]] = {name: [] for name in REGION_NAMES} if return_activity else {}

        def silence(name: str, *tensors: Tensor) -> tuple[Tensor, ...]:
            if name in disabled:
                return tuple(torch.zeros_like(tensor) for tensor in tensors)
            return tensors

        r = self.regions
        for t in range(length):
            visual, visual_logits = silence("visual_cortex", *r["visual_cortex"](observations["visual"][:, t], previous_prefrontal))
            auditory, auditory_logits = silence("auditory_cortex", *r["auditory_cortex"](observations["auditory"][:, t], previous_prefrontal))
            (body,) = silence("somatosensory_cortex", r["somatosensory_cortex"](observations["body"][:, t], observations["feedback"][:, t], previous_motor))
            language = r["temporal_language"](observations["tokens"][:, t], previous_prefrontal)
            if language_context is not None:
                # Residual input preserves the original computation for zero
                # context as well as None; temporal lesions silence both paths.
                language = language + language_context[:, t]
            (language,) = silence("temporal_language", language)
            (salience,) = silence("amygdala", r["amygdala"](visual, auditory, body))
            (arousal,) = silence("hypothalamus", r["hypothalamus"](body))
            (association,) = silence("parietal_association", r["parietal_association"](visual, auditory, body, language))
            sources = torch.stack((visual, auditory, body, language, association), dim=1)
            (routed,) = silence("thalamus", r["thalamus"](sources, previous_prefrontal, language, salience, arousal))
            if "hippocampus" in disabled:
                # Neither hidden writes nor reads survive a hippocampal ablation.
                recalled, memory = zero, ()
            else:
                recalled, memory = r["hippocampus"](
                    association, language, body, previous_prefrontal, salience, memory,
                    external_context=None if memory_context is None else memory_context[:, t],
                )
            (prefrontal,) = silence("prefrontal_cortex", r["prefrontal_cortex"](routed, recalled, language, previous_cerebellum, salience, arousal, previous_prefrontal))
            selection, gate, value = silence("basal_ganglia", *r["basal_ganglia"](prefrontal, recalled, salience))
            motor, logits = silence("motor_cortex", *r["motor_cortex"](prefrontal, routed, selection, previous_cerebellum, gate))
            actual = torch.cat((observations["visual"][:, t], observations["auditory"][:, t], observations["body"][:, t]), dim=-1)
            prediction_error = actual - previous_prediction
            cerebellum, prediction = silence("cerebellum", *r["cerebellum"](prefrontal, motor, body, prediction_error))

            for key, tensor in (
                ("logits", logits), ("visual_logits", visual_logits),
                ("auditory_logits", auditory_logits), ("prediction", prediction), ("value", value),
            ):
                outputs[key].append(tensor)
            if return_activity:
                current = {
                    "visual_cortex": visual, "auditory_cortex": auditory,
                    "somatosensory_cortex": body, "temporal_language": language,
                    "parietal_association": association, "thalamus": routed,
                    "prefrontal_cortex": prefrontal, "hippocampus": recalled,
                    "basal_ganglia": selection, "motor_cortex": motor,
                    "cerebellum": cerebellum, "amygdala": salience, "hypothalamus": arousal,
                }
                for name, tensor in current.items():
                    activities[name].append(tensor)
            previous_prefrontal, previous_motor = prefrontal, motor
            previous_cerebellum, previous_prediction = cerebellum, prediction

        result: dict[str, Tensor | dict[str, Tensor]] = {key: torch.stack(values, dim=1) for key, values in outputs.items()}
        if return_activity:
            result["region_activity"] = {name: torch.stack(values, dim=1) for name, values in activities.items()}
        final_state = BrainState(previous_prefrontal, previous_motor, previous_cerebellum,
                                 previous_prediction, memory)
        return result, final_state
