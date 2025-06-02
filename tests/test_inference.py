import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from beast.inference import ImagePredictionHandler, VideoPredictionHandler


class TestImagePredictionHandler:
    """Test suite for PredictionHandler class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary source and output directories."""
        temp_source = Path(tempfile.mkdtemp())
        temp_output = Path(tempfile.mkdtemp())

        # Create test directory structure
        (temp_source / "video1").mkdir(parents=True)
        (temp_source / "video2").mkdir(parents=True)

        # Create dummy image files
        for video in ["video1", "video2"]:
            for frame in ["frame001.png", "frame002.png"]:
                dummy_img = Image.new('RGB', (64, 64), color='red')
                dummy_img.save(temp_source / video / frame)

        yield temp_source, temp_output

        # Cleanup
        shutil.rmtree(temp_source)
        shutil.rmtree(temp_output)

    @pytest.fixture
    def handler(self, temp_dirs):
        """Create PredictionHandler instance with temp directories."""
        source_dir, output_dir = temp_dirs
        return ImagePredictionHandler(output_dir, source_dir)

    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for testing."""
        return torch.rand(3, 64, 64)  # (C, H, W) format

    @pytest.fixture
    def sample_batch_tensor(self):
        """Create sample batch tensor for testing."""
        return torch.rand(2, 3, 64, 64)  # (B, C, H, W) format

    @pytest.fixture
    def sample_latents(self):
        """Create sample latent tensor."""
        return torch.rand(2, 128)  # (B, latent_dim)

    @pytest.fixture
    def sample_metadata(self, temp_dirs):
        """Create sample batch metadata."""
        source_dir, _ = temp_dirs
        return {
            'video': ['video1', 'video1'],
            'idx': [torch.tensor(0), torch.tensor(1)],
            'image_paths': [
                source_dir / "video1" / "frame001.png",
                source_dir / "video1" / "frame002.png"
            ]
        }

    def test_init(self, temp_dirs):
        """Test PredictionHandler initialization."""
        source_dir, output_dir = temp_dirs
        handler = ImagePredictionHandler(output_dir, source_dir)
        assert handler.output_dir == Path(output_dir)
        assert handler.source_dir == Path(source_dir)
        assert handler.output_dir.exists()
        assert handler.metadata == []

    def test_tensor_to_image_3d(self, handler, sample_tensor):
        """Test tensor to image conversion with 3D tensor (C, H, W)."""
        image = handler.tensor_to_image(sample_tensor)
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (64, 64)

    def test_tensor_to_image_4d(self, handler, sample_batch_tensor):
        """Test tensor to image conversion with 4D tensor (B, C, H, W)."""
        image = handler.tensor_to_image(sample_batch_tensor)
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (64, 64)

    def test_tensor_to_image_grayscale(self, handler):
        """Test tensor to image conversion with grayscale (1 channel)."""
        tensor_gray = torch.rand(1, 32, 32)
        image = handler.tensor_to_image(tensor_gray)
        assert isinstance(image, Image.Image)
        assert image.mode == 'L'  # Grayscale mode
        assert image.size == (32, 32)

    def test_tensor_to_image_scaling(self, handler):
        """Test tensor value scaling from [0,1] to [0,255]."""
        # Create tensor with known values
        tensor = torch.ones(3, 2, 2) * 0.5  # All values = 0.5
        image = handler.tensor_to_image(tensor)
        # Check that values were scaled (0.5 * 255 = 127.5 → 127)
        np_array = np.array(image)
        assert np_array.max() <= 255
        assert np_array.min() >= 1

    def test_save_reconstruction(self, handler, sample_tensor, temp_dirs):
        """Test saving reconstruction image."""
        source_dir, output_dir = temp_dirs
        original_path = source_dir / 'video1' / 'frame001.png'

        saved_path = handler.save_reconstruction(
            sample_tensor, 'video1', 0, original_path
        )

        expected_path = output_dir / 'video1' / 'frame001.png'
        assert saved_path == expected_path
        assert saved_path.exists()
        assert saved_path.is_file()
        assert (output_dir / 'video1').exists()

    def test_save_latents(self, handler, temp_dirs):
        """Test saving latent representations."""
        source_dir, output_dir = temp_dirs
        original_path = source_dir / 'video1' / 'frame001.png'
        latents = torch.rand(128)

        saved_path = handler.save_latents(latents, 'video1', 0, original_path)

        expected_path = output_dir / 'latents' / 'video1' / 'frame001.npy'
        assert saved_path == expected_path
        assert saved_path.exists()
        assert saved_path.is_file()
        assert (output_dir / 'latents' / 'video1').exists()

        # Check that saved data matches
        loaded_latents = np.load(saved_path)
        np.testing.assert_array_almost_equal(loaded_latents, latents.detach().cpu().numpy())

    def test_process_batch_predictions_reconstructions_only(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata,
    ):
        """Test processing batch with reconstructions only."""
        predictions = {
            'reconstructions': sample_batch_tensor,
            'latents': sample_latents
        }

        result = handler.process_batch_predictions(
            predictions,
            sample_metadata,
            save_reconstructions=True,
            save_latents=False
        )

        # Check results structure
        assert 'reconstructions' in result
        assert 'latents' in result
        assert 'metadata' in result

        # Should have saved reconstructions
        assert len(result['reconstructions']) == 2
        assert len(result['latents']) == 0  # No latents saved
        assert len(result['metadata']) == 2

        # Check metadata entries
        for i, metadata in enumerate(result['metadata']):
            assert 'original_path' in metadata
            assert 'video' in metadata
            assert 'idx' in metadata
            assert 'reconstruction_path' in metadata
            assert 'latents_path' not in metadata  # Latents not saved

    def test_process_batch_predictions_latents_only(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata
    ):
        """Test processing batch with latents only."""
        predictions = {
            'reconstructions': sample_batch_tensor,
            'latents': sample_latents
        }

        result = handler.process_batch_predictions(
            predictions,
            sample_metadata,
            save_reconstructions=False,
            save_latents=True
        )

        # Should have saved latents only
        assert len(result['reconstructions']) == 0
        assert len(result['latents']) == 2
        assert len(result['metadata']) == 2

        # Check metadata entries
        for metadata in result['metadata']:
            assert 'reconstruction_path' not in metadata
            assert 'latents_path' in metadata

    def test_process_batch_predictions_both(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata,
    ):
        """Test processing batch with both reconstructions and latents."""
        predictions = {
            'reconstructions': sample_batch_tensor,
            'latents': sample_latents
        }

        result = handler.process_batch_predictions(
            predictions,
            sample_metadata,
            save_reconstructions=True,
            save_latents=True
        )

        # Should have saved both
        assert len(result['reconstructions']) == 2
        assert len(result['latents']) == 2
        assert len(result['metadata']) == 2

        # Check metadata entries have both paths
        for metadata in result['metadata']:
            assert 'reconstruction_path' in metadata
            assert 'latents_path' in metadata

    def test_save_metadata_summary(self, handler, temp_dirs):
        """Test saving metadata summary to YAML."""
        # Add some test metadata
        test_metadata = [
            {
                'original_path': '/path/to/original.png',
                'reconstruction_path': '/path/to/recon.png',
                'video': 'video1',
                'idx': 0
            }
        ]
        handler.metadata = test_metadata

        metadata_path = handler.save_metadata_summary()

        _, output_dir = temp_dirs
        expected_path = output_dir / 'prediction_metadata.yaml'

        assert metadata_path == expected_path
        assert metadata_path.exists()

        # Check YAML content
        with open(metadata_path, 'r') as f:
            loaded_metadata = yaml.safe_load(f)

        assert loaded_metadata == test_metadata

    def test_process_predictions_full_workflow(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata
    ):
        """Test full workflow with process_predictions method."""
        # Create mock predictions (list of batches)
        predictions = [
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
                'metadata': sample_metadata
            }
        ]

        result = handler.process_predictions(
            predictions,
            save_reconstructions=True,
            save_latents=True
        )

        # Check results structure
        assert 'output_dir' in result
        assert 'num_images_processed' in result
        assert 'metadata_file' in result
        assert 'reconstructions_saved' in result
        assert 'latents_saved' in result
        assert 'reconstructions_dir' in result
        assert 'latents_dir' in result

        # Check counts
        assert result['num_images_processed'] == 2
        assert result['reconstructions_saved'] == 2
        assert result['latents_saved'] == 2

        # Check metadata file was created
        metadata_path = Path(result['metadata_file'])
        assert metadata_path.exists()

    def test_process_predictions_empty_list(self, handler):
        """Test process_predictions with empty prediction list."""
        result = handler.process_predictions([], save_reconstructions=True)

        assert result['num_images_processed'] == 0
        assert result['reconstructions_saved'] == 0

    def test_directory_creation(self, handler, sample_tensor, temp_dirs):
        """Test that subdirectories are created properly."""
        source_dir, output_dir = temp_dirs
        original_path = source_dir / 'new_video' / 'frame001.png'

        # This should create the new_video directory
        saved_path = handler.save_reconstruction(
            sample_tensor, 'new_video', 0, original_path
        )

        assert (output_dir / 'new_video').exists()
        assert saved_path.parent == output_dir / 'new_video'

    @pytest.mark.parametrize('save_recons,save_latents', [
        (True, False),
        (False, True),
        (True, True),
        (False, False)
    ])
    def test_process_predictions_save_options(
            self, handler, sample_batch_tensor, sample_latents, sample_metadata,
            save_recons, save_latents
    ):
        """Test different combinations of save options."""
        predictions = [
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
                'metadata': sample_metadata
            }
        ]

        result = handler.process_predictions(
            predictions,
            save_reconstructions=save_recons,
            save_latents=save_latents
        )

        if save_recons:
            assert 'reconstructions_saved' in result
            assert result['reconstructions_saved'] == 2
        else:
            assert result.get('reconstructions_saved', 0) == 0

        if save_latents:
            assert 'latents_saved' in result
            assert result['latents_saved'] == 2
        else:
            assert result.get('latents_saved', 0) == 0


class TestVideoPredictionHandler:
    """Test suite for VideoPredictionHandler class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary source and output directories."""
        temp_source = Path(tempfile.mkdtemp())
        temp_output = Path(tempfile.mkdtemp())

        # create test video file
        video_file = temp_source / 'video1.mp4'
        # create a simple test video with 10 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (64, 64))
        for i in range(10):
            # create frames with different colors
            frame = np.full((64, 64, 3), (i * 25, i * 25, i * 25), dtype=np.uint8)
            out.write(frame)

        out.release()

        yield temp_source, temp_output, video_file

        # cleanup
        shutil.rmtree(temp_source)
        shutil.rmtree(temp_output)

    @pytest.fixture
    def handler(self, temp_dirs):
        """Create VideoPredictionHandler instance with temp directories."""
        source_dir, output_dir, video_file = temp_dirs
        return VideoPredictionHandler(output_dir, video_file)

    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for testing."""
        return torch.rand(3, 64, 64)  # (C, H, W) format

    @pytest.fixture
    def sample_batch_tensor(self):
        """Create sample batch tensor for testing."""
        return torch.rand(2, 3, 64, 64)  # (B, C, H, W) format

    @pytest.fixture
    def sample_latents(self):
        """Create sample latent tensor."""
        return torch.rand(2, 128)  # (B, latent_dim)

    @pytest.fixture
    def sample_metadata(self, temp_dirs):
        """Create sample batch metadata for video."""
        source_dir, _, video_paths = temp_dirs
        return {
            'video_path': [str(video_paths[0]), str(video_paths[0])],
            'frame_idx': [torch.tensor(0), torch.tensor(1)],
            'batch_start_idx': torch.tensor(0)
        }

    def test_init(self, temp_dirs):
        """Test VideoPredictionHandler initialization."""
        _, output_dir, video_file = temp_dirs
        handler = VideoPredictionHandler(output_dir, video_file)
        assert handler.output_dir == Path(output_dir)
        assert handler.output_dir.exists()
        assert handler.metadata == {
            'video_file': str(video_file),
            'output_dir': str(output_dir),
            'fps': 10,
            'width': 64,
            'height': 64,
            'total_frames': 10,
        }

    def test_tensor_to_image_3d(self, handler, sample_tensor):
        """Test tensor to image conversion with 3D tensor (C, H, W)."""
        image = handler.tensor_to_numpy_bgr(sample_tensor)
        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 3)

    def test_tensor_to_image_4d(self, handler, sample_batch_tensor):
        """Test tensor to image conversion with 4D tensor (B, C, H, W)."""
        image = handler.tensor_to_numpy_bgr(sample_batch_tensor)
        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 3)

    def test_tensor_to_image_grayscale(self, handler):
        """Test tensor to image conversion with grayscale (1 channel)."""
        tensor_gray = torch.rand(1, 32, 32)
        image = handler.tensor_to_numpy_bgr(tensor_gray)
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)

    def test_tensor_to_image_scaling(self, handler):
        """Test tensor value scaling from [0,1] to [0,255]."""
        # Create tensor with known values
        tensor = torch.ones(3, 2, 2) * 0.5  # All values = 0.5
        image = handler.tensor_to_numpy_bgr(tensor)
        # Check that values were scaled (0.5 * 255 = 127.5 → 127)
        np_array = np.array(image)
        assert np_array.max() <= 255
        assert np_array.min() >= 1

    def test_process_predictions_empty_list(self, handler):
        """Test process_predictions with empty prediction list."""
        result = handler.process_predictions([], save_reconstructions=True)

        assert result['frames_processed'] == 0
        assert result['reconstruction_video'] is None

    @pytest.mark.parametrize('save_recons,save_latents', [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ])
    def test_process_predictions_save_options(
        self, handler, sample_batch_tensor, sample_latents, save_recons, save_latents,
    ):
        """Test different combinations of save options."""

        predictions = [
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
            },
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
            },
        ]

        result = handler.process_predictions(
            predictions,
            save_reconstructions=save_recons,
            save_latents=save_latents,
        )

        if save_recons:
            assert Path(result['reconstruction_video']).is_file()
        else:
            assert result['reconstruction_video'] is None

        if save_latents:
            assert Path(result['latents_file']).is_file()
            assert result['latents_shape'] == (4, 128)
        else:
            assert result['latents_file'] is None
