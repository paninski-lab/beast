import copy

import pytest


def test_load_config(config_ae_path):

    from beast.io import load_config

    # test normal loading
    load_config(config_ae_path)

    # test Assertion error with bad path
    with pytest.raises(AssertionError):
        load_config('/fake/path')


def test_apply_config_overrides(config_ae):

    from beast.io import apply_config_overrides

    # override existing fields with dict
    overrides = {'model.seed': 1}
    new_config = apply_config_overrides(copy.deepcopy(config_ae), overrides)
    assert new_config['model']['seed'] == overrides['model.seed']

    # add new fields at different levels with dict
    overrides = {'data': '/path/to/data', 'model.seed': 2, 'model.model_params.batchnorm': True}
    new_config = apply_config_overrides(copy.deepcopy(config_ae), overrides)
    assert new_config['data'] == overrides['data']
    assert new_config['model']['seed'] == overrides['model.seed']
    assert new_config['model']['model_params']['batchnorm'] == overrides[
        'model.model_params.batchnorm'
    ]

    # override existing fields with list
    overrides = ['model.seed=1', 'training.imgaug=geometric']
    new_config = apply_config_overrides(copy.deepcopy(config_ae), overrides)
    assert new_config['model']['seed'] == '1'
    assert new_config['training']['imgaug'] == 'geometric'
