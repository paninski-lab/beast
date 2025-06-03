"""Command to run model inference on videos."""

import logging
from pathlib import Path

_logger = logging.getLogger('BEAST.CLI.PREDICT')


def register_parser(subparsers):
    """Register the predict command parser."""

    parser = subparsers.add_parser(
        'predict',
        description='Run inference using a trained model on image or video data.',
        usage='beast predict --model <model_dir> --input <video_path> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--model', '-m',
        type=Path,
        required=True,
        help='Directory containing trained model',
    )
    required.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to input video file or directory of images or videos',
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--output', '-o',
        type=Path,
        help='Directory to save prediction results (default: <model_dir>/predictions)',
    )
    optional.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)',
    )
    optional.add_argument(
        '--save_latents', '-l',
        action='store_true',
        help='Extract and save latent features',
    )
    optional.add_argument(
        '--save_reconstructions', '-r',
        action='store_true',
        help='Extract and save reconstructions',
    )


def handle(args):
    """Handle the predict command execution."""

    _logger.info(f'Running inference with model from: {args.model}')
    _logger.info(f'Input: {args.input}')
    _logger.info(f'Output directory: {args.output or args.model}')
    if not args.save_latents and not args.save_reconstructions:
        _logger.warning(
            f'did not detect --save_latents or --save_reconstructions; no outputs will be saved'
        )

    # Load model
    from beast.api.model import Model
    model = Model.from_dir(args.model)

    # Run prediction
    if args.input.is_file():
        # Single video inference
        model.predict_video(
            video_file=args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
            save_latents=True,
            save_reconstructions=args.save_reconstructions,
        )

    elif args.input.is_dir():

        videos = list(args.input.rglob('*.mp4')) + list(args.input.rglob('*.avi'))
        num_videos = len(videos)

        num_images = len(
            list(args.input.rglob('*.png'))
            + list(args.input.rglob('*.jpg'))
            + list(args.input.rglob('*.jpeg'))
        )

        if num_videos > 0 and num_images > 0:
            _logger.error(f'Found both videos and images in {args.input}; aborting')
            return
        elif num_videos > 0:
            for video_file in videos:
                _logger.info(f'Running inference on {video_file}')
                model.predict_video(
                    video_file=video_file,
                    output_dir=args.output,
                    batch_size=args.batch_size,
                    save_latents=True,
                    save_reconstructions=args.save_reconstructions,
                )
        else:
            model.predict_images(
                image_dir=args.input,
                output_dir=args.output,
                batch_size=args.batch_size,
                save_latents=args.save_latents,
                save_reconstructions=args.save_reconstructions,
            )
