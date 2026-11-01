"""Tests for BaseLightningModel."""

from types import SimpleNamespace

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


class TestGetScheduler:
    """Test BaseLightningModel.get_scheduler."""

    @staticmethod
    def _make_scheduler(accumulate: int, final_div_factor: float = 1000.0):
        config = {
            'training': {
                'train_batch_size': 128,
                'num_gpus': 2,
                'num_nodes': 1,
                'num_epochs': 10,
            },
            'optimizer': {
                'scheduler': 'cosine',
                'lr': 1e-4,
                'accumulate_grad_batches': accumulate,
                'warmup_pct': 0.15,
                'div_factor': 10.0,
                'final_div_factor': final_div_factor,
            },
        }
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = BaseLightningModel.get_scheduler(SimpleNamespace(config=config), optimizer)
        return scheduler

    def test_get_scheduler_peak_lr_scales_with_batch(self):
        scheduler = self._make_scheduler(accumulate=1)
        # 128 * 2 gpus = 256 -> peak equals base lr
        assert scheduler.optimizer.param_groups[0]['max_lr'] == (
            pytest.approx(1e-4)
        )

    def test_get_scheduler_peak_lr_includes_accumulation(self):
        scheduler = self._make_scheduler(accumulate=4)
        assert scheduler.optimizer.param_groups[0]['max_lr'] == pytest.approx(4e-4)

    def test_get_scheduler_final_lr_uses_final_div_factor(self):
        scheduler = self._make_scheduler(accumulate=1)
        assert scheduler.optimizer.param_groups[0]['min_lr'] == pytest.approx(1e-4 / 10 / 1000)
