"""Command to run the BEAST3D preprocessing pipeline."""

import argparse
import logging
from typing import Any

from beast.cli.types import config_file

_logger = logging.getLogger('BEAST.CLI.EXTRACT_3D')


def register_parser(subparsers: Any) -> None:
    """Register the extract_3d command parser."""

    parser = subparsers.add_parser(
        'extract_3d',
        description=(
            'Run the BEAST3D preprocessing pipeline '
            '(stats, trim, downsample, segment, assemble, resize).'
        ),
        usage='beast extract_3d --config <config_path> [options]',
    )

    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--config', '-c',
        type=config_file,
        required=True,
        help='Path to BEAST3D config file (YAML)',
    )

    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--skip-stats',
        action='store_true',
        default=False,
        help='Skip the video stats step',
    )


def handle(args: argparse.Namespace) -> None:
    """Handle the extract_3d command execution."""

    from beast.preprocess.config_3d import load_config_3d, validate_config
    from beast.preprocess.extraction_3d import (
        assemble_dataset,
        resize_dataset,
        resolve_videos_dir,
        run_downsample,
        run_trim,
        run_video_stats,
    )
    from beast.preprocess.segment.pipeline import run_segmentation

    cfg = load_config_3d(args.config)
    validate_config(cfg)

    if not args.skip_stats:
        _logger.info('step: video stats')
        run_video_stats(cfg)

    if cfg.cut.enabled:
        _logger.info('step: trim')
        run_trim(cfg)

    if cfg.downsample.enabled:
        _logger.info('step: downsample')
        run_downsample(cfg)

    if cfg.segmentation.enabled:
        _logger.info('step: segment')
        videos_dir = resolve_videos_dir(cfg)
        run_segmentation(videos_dir, cfg)

    _logger.info('step: assemble')
    assemble_dataset(cfg)

    if cfg.resize.enabled:
        _logger.info('step: resize')
        resize_dataset(cfg)

    _logger.info('extract_3d pipeline complete')
