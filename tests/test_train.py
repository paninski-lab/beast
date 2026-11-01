"""Tests for training helper functions."""

import os
import random

import numpy as np
import torch

from beast.train import reset_seeds


class TestResetSeeds:
    """Test the reset_seeds function."""

    def test_sets_pythonhashseed_env_var(self) -> None:
        reset_seeds(42)
        assert os.environ['PYTHONHASHSEED'] == '42'

    def test_torch_produces_same_values_after_reset(self) -> None:
        reset_seeds(0)
        a = torch.randn(5)
        reset_seeds(0)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_numpy_produces_same_values_after_reset(self) -> None:
        reset_seeds(0)
        a = np.random.rand(5)
        reset_seeds(0)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_python_random_produces_same_values_after_reset(self) -> None:
        reset_seeds(0)
        a = [random.random() for _ in range(5)]
        reset_seeds(0)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_cudnn_flags_set(self) -> None:
        reset_seeds(0)
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
