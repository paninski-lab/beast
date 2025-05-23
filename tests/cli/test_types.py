from pathlib import Path

import pytest


def test_valid_file(tmpdir):

    from beast.cli.types import valid_file

    tmpdir = Path(tmpdir)

    # file exists
    new_file_0 = tmpdir.joinpath('test_file_0.yaml')
    new_file_0.touch()
    new_file_out = valid_file(new_file_0)
    assert new_file_out.is_file()

    # file does not exist
    new_file_1 = tmpdir.joinpath('test_file_1.yaml')
    with pytest.raises(IOError):
        valid_file(new_file_1)

    # file is a directory
    new_dir = tmpdir.joinpath('test_dir')
    with pytest.raises(IOError):
        valid_file(new_dir)


def test_valid_dir(tmpdir):

    from beast.cli.types import valid_dir

    tmpdir = Path(tmpdir)

    # dir exists
    new_dir_0 = tmpdir.joinpath('new_dir_0')
    new_dir_0.mkdir(parents=True, exist_ok=True)
    new_dir_out = valid_dir(new_dir_0)
    assert new_dir_out.is_dir()

    # dir does not exist
    new_dir_1 = tmpdir.joinpath('new_dir_1')
    with pytest.raises(IOError):
        valid_dir(new_dir_1)

    # pass in a file
    new_file_0 = tmpdir.joinpath('test_file_0.yaml')
    new_file_0.touch()
    with pytest.raises(IOError):
        valid_dir(new_file_0)


def test_config_file(tmpdir):

    from beast.cli.types import config_file

    tmpdir = Path(tmpdir)

    # file exists and is of proper extension
    new_file_0 = tmpdir.joinpath('config_0.yaml')
    new_file_0.touch()
    new_file_out = config_file(new_file_0)
    assert new_file_out.is_file()

    new_file_1 = tmpdir.joinpath('config_1.yml')
    new_file_1.touch()
    new_file_out = config_file(new_file_1)
    assert new_file_out.is_file()

    # file exists but is not proper extension
    new_file_2 = tmpdir.joinpath('config_2.json')
    new_file_2.touch()
    with pytest.raises(ValueError):
        config_file(new_file_2)


def test_output_dir(tmpdir):

    from beast.cli.types import output_dir

    tmpdir = Path(tmpdir)

    out_dir = output_dir(tmpdir.joinpath('test0'))
    assert out_dir.is_dir()

    out_dir = output_dir(tmpdir.joinpath('test1').joinpath('test2'))
    assert out_dir.is_dir()
