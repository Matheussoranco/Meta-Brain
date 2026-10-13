from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from metalearning.algorithms import create_differentiable_snn_from_connectome

CONFIG = str(Path(__file__).resolve().parents[1] / 'config' / 'config.yaml')


def test_full_connectome_refused_before_matrix_allocation(monkeypatch):
    graph = SimpleNamespace(neurons=dict.fromkeys(range(139000)), edges=[])
    monkeypatch.setattr(torch, 'zeros', lambda *args, **kwargs: pytest.fail('Unsafe dense allocation'))
    with pytest.raises(ValueError, match='Dense SNN requires'):
        create_differentiable_snn_from_connectome(graph, [], [], CONFIG)


def test_small_subgraph_remains_differentiable():
    graph = SimpleNamespace(neurons={
        1: SimpleNamespace(neurotransmitter='acetylcholine'),
        2: SimpleNamespace(neurotransmitter='gaba'),
    }, edges=[(1, 2, 2)])
    snn = create_differentiable_snn_from_connectome(graph, [1], [2], CONFIG)
    assert snn.connectivity.shape == (2, 2)
    assert torch.isclose(snn.connectivity[1, 0], torch.tensor(1.0))
    output, _ = snn(torch.ones(3, 1, 1))
    output.sum().backward()
    assert snn.W_syn.grad is not None


@pytest.mark.parametrize('budget', [0, -1, True, '1024'])
def test_invalid_budget_rejected(budget):
    graph = SimpleNamespace(neurons={1: SimpleNamespace(neurotransmitter='gaba')}, edges=[])
    with pytest.raises(ValueError, match='positive integer'):
        create_differentiable_snn_from_connectome(graph, [1], [1], CONFIG, budget)
