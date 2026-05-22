"""Tests for BaseLightningModel."""

import pytest
import torch

from beast.models.base import BaseLightningModel


class TestBaseLightningModelAbstract:
    """Test that BaseLightningModel enforces its abstract interface."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseLightningModel({'model': {'seed': 0}})  # type: ignore[abstract]

    def test_missing_get_model_outputs_raises(self) -> None:
        class Incomplete(BaseLightningModel):
            def compute_loss(self, stage, **kwargs):
                pass

            def predict_step(self, batch_dict, batch_idx):
                pass

        with pytest.raises(TypeError):
            Incomplete({'model': {'seed': 0}})  # type: ignore[abstract]

    def test_missing_compute_loss_raises(self) -> None:
        class Incomplete(BaseLightningModel):
            def get_model_outputs(self, batch_dict):
                pass

            def predict_step(self, batch_dict, batch_idx):
                pass

        with pytest.raises(TypeError):
            Incomplete({'model': {'seed': 0}})  # type: ignore[abstract]

    def test_missing_predict_step_raises(self) -> None:
        class Incomplete(BaseLightningModel):
            def get_model_outputs(self, batch_dict):
                pass

            def compute_loss(self, stage, **kwargs):
                pass

        with pytest.raises(TypeError):
            Incomplete({'model': {'seed': 0}})  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        class Concrete(BaseLightningModel):
            def get_model_outputs(self, batch_dict):
                return {}

            def compute_loss(self, stage, **kwargs):
                return torch.tensor(0.0), []

            def predict_step(self, batch_dict, batch_idx):
                return {}

        model = Concrete({'model': {'seed': 0}})
        assert isinstance(model, BaseLightningModel)
