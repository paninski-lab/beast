"""Command to extract frames from videos."""

import logging
from pathlib import Path

from beast.cli.types import output_dir

_logger = logging.getLogger('BEAST.CLI.EXTRACT')


def register_parser(subparsers):
    """Register the extract command parser."""

    parser = subparsers.add_parser(
        'extract',
        description='Extract frames from videos for model training.',
        usage='beast extract --input <video_dir> --output <output_dir> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Directory containing video files or path to a single video',
    )
    required.add_argument(
        '--output', '-o',
        type=output_dir,
        required=True,
        help='Directory to save extracted frames',
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--frames-per-video', '-n',
        type=int,
        default=500,
        help='Number of frames to extract from each video (default: 500)',
    )
    optional.add_argument(
        '--method', '-m',
        choices=['uniform', 'random', 'pca_kmeans'],
        default='pca_kmeans',
        help='Frame extraction method (default: pca_kmeans)',
    )
    optional.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of worker processes (default: 4)',
    )


def handle(args):
    """Handle the extract command execution."""

    _logger.info(f'Running frame extraction with {args.method} method')
    _logger.info(f'Input directory: {args.input}')
    _logger.info(f'Output directory: {args.output}')

    # Create output directory if it doesn't exist
    args.output.mkdir(parents=True, exist_ok=True)

    # Import the actual implementation
    from beast.extraction import extract_frames

    # Call the implementation
    result = extract_frames(
        input_path=args.input,
        output_dir=args.output,
        frames_per_video=args.frames_per_video,
        method=args.method,
        num_workers=args.workers,
    )

    # Print summary
    _logger.info(f'Extracted {result["total_frames"]} frames from {result["total_videos"]} videos')
