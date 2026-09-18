"""Optional feedback-learning agents; model outputs denote object categories.

The symbolic category-to-cell mapping is explicit, not learned perception.
The neural policy learns a strategy offline and adapts its state, not weights,
during use. Nothing in this module changes the released desktop controller.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import torch
from torch import nn

from .adaptation import encode_observations
from .model import Brain, BrainConfig


def brain_config(hidden_size=16):
    return BrainConfig(hidden_size=hidden_size, visual_dim=16, auditory_dim=4,
                       body_dim=4, vocab_size=1, num_actions=4, memory_slots=8)


class FlatGRU(nn.Module):
    """A recurrent reference with the same observable inputs as BiC."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(26, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 4)

    def forward_with_state(self, observations, state=None):
        inputs = torch.cat([observations[k] for k in
                            ('visual', 'auditory', 'body', 'feedback')], dim=-1)
        activity, state = self.gru(inputs, state)
        return {'logits': self.head(activity)}, state

    def forward(self, observations):
        return self.forward_with_state(observations)[0]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def build_model(kind='bic', hidden_size=16):
    if kind == 'bic':
        return Brain(brain_config(hidden_size))
    if kind != 'gru':
        raise ValueError('kind must be bic or gru')
    target = parameter_count(Brain(brain_config(hidden_size)))
    # GRU: 3H*26 + 3H*H + 6H, followed by 4H+4 classifier.
    width = min(range(1, 512), key=lambda h: abs(3*h*h + 88*h + 4 - target))
    return FlatGRU(width)


def model_digest(model):
    digest = hashlib.sha256()
    config = asdict(model.config) if isinstance(model, Brain) else {'hidden_size': model.hidden_size}
    digest.update(json.dumps({'kind': type(model).__name__, 'config': config}, sort_keys=True).encode('utf-8'))
    for name, value in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_state(model, state):
    if state is None:
        return None
    if isinstance(model, Brain):
        return model.state_to_dict(state)
    return state.detach().cpu().clone()


def import_state(model, payload):
    if payload is None:
        return None
    if isinstance(model, Brain):
        return model.state_from_dict(payload)
    reference = next(model.parameters())
    if (not isinstance(payload, torch.Tensor) or payload.ndim != 3
            or payload.shape[0] != 1 or payload.shape[2] != model.hidden_size
            or not torch.isfinite(payload).all()):
        raise ValueError('invalid GRU state')
    return payload.detach().to(device=reference.device, dtype=reference.dtype).clone()


class AdaptiveSession:
    """One explicit stream of experience with save/resume support."""

    def __init__(self, model):
        self.model = model.eval()
        self.state = None
        self.steps = 0

    @torch.inference_mode()
    def act(self, observation):
        device = next(self.model.parameters()).device
        encoded = encode_observations([observation], device=device)
        output, self.state = self.model.forward_with_state(encoded, self.state)
        category = int(output['logits'][0, -1].argmax())
        self.steps += 1
        return observation['layout'].index(category)

    def snapshot(self):
        return {'schema': 1, 'model_sha256': model_digest(self.model),
                'steps': self.steps, 'state': export_state(self.model, self.state)}

    def restore(self, snapshot):
        if (snapshot.get('schema') != 1
                or snapshot.get('model_sha256') != model_digest(self.model)
                or type(snapshot.get('steps')) is not int or snapshot['steps'] < 0):
            raise ValueError('session does not match this model or has invalid metadata')
        state = import_state(self.model, snapshot['state'])
        if (state is None) != (snapshot['steps'] == 0):
            raise ValueError('session state and step count are inconsistent')
        if state is not None:
            batch = state.previous_prefrontal.shape[0] if isinstance(self.model, Brain) else state.shape[1]
            if batch != 1:
                raise ValueError('a session requires exactly one stream')
        self.state, self.steps = state, snapshot['steps']


def save_agent(model, path, *, seed, hidden_size, metadata=None):
    kind = 'bic' if isinstance(model, Brain) else 'gru'
    torch.save({'schema': 'bic-adaptation-1', 'kind': kind,
                'brain_hidden_size': hidden_size, 'seed': seed,
                'parameters': parameter_count(model), 'state_dict': model.state_dict(),
                'metadata': metadata or {}}, path)


def load_agent(path, device='cpu'):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('schema') != 'bic-adaptation-1':
        raise ValueError('unsupported adaptation checkpoint')
    model = build_model(data['kind'], data['brain_hidden_size'])
    model.load_state_dict(data['state_dict'], strict=True)
    return model.to(device).eval(), data
