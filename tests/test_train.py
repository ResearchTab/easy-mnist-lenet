from __future__ import annotations

import torch

from train import (
    EXPECTED_TRAINABLE_PARAMETERS,
    SPLIT_SEED,
    LeNet5,
    ParameterMatchedMLP,
    class_balanced_split,
    create_model,
    trainable_parameter_count,
)


def test_lenet_output_shape() -> None:
    output = LeNet5()(torch.zeros(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_parameter_matched_models_have_identical_capacity_and_output_shape() -> None:
    lenet = create_model("lenet")
    mlp = create_model("mlp-matched")
    assert isinstance(lenet, LeNet5)
    assert isinstance(mlp, ParameterMatchedMLP)
    assert trainable_parameter_count(lenet) == EXPECTED_TRAINABLE_PARAMETERS
    assert trainable_parameter_count(mlp) == EXPECTED_TRAINABLE_PARAMETERS
    assert mlp(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_class_balanced_split_is_deterministic_and_disjoint() -> None:
    targets = torch.arange(10).repeat_interleave(6_000)
    first_train, first_validation = class_balanced_split(targets, SPLIT_SEED)
    second_train, second_validation = class_balanced_split(targets, SPLIT_SEED)
    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 40_000
    assert len(first_validation) == 10_000
    assert set(first_train).isdisjoint(first_validation)
    for digit in range(10):
        assert sum(int(targets[index]) == digit for index in first_train) == 4_000
        assert sum(int(targets[index]) == digit for index in first_validation) == 1_000
