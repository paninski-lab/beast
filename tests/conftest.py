from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture
def video_file() -> Path:
    return ROOT.joinpath('tests/data/test_vid.mp4')
