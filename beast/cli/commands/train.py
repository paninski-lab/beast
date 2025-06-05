"""Command to train a model."""

import datetime
import logging
from pathlib import Path

from beast.cli.types import config_file, output_dir

_logger = logging.getLogger('BEAST.CLI.TRAIN')


def register_parser(subparsers):
    """Register the train command parser."""

    parser = subparsers.add_parser(
        'train',
        description='Train a neural network model on video frame data.',
        usage='beast train --config <config_path> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--config', '-c',
        type=config_file,
        required=True,
        help='Path to model configuration file (YAML)',
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--output', '-o',
        type=output_dir,
        help='Directory to save model outputs (default: ./runs/YYYY-MM-DD/HH-MM-SS)',
    )
    optional.add_argument(
        '--data', '-d',
        type=Path,
        help='Override data directory specified in config',
    )
    optional.add_argument(
        '--gpus',
        type=int,
        help='Number of GPUs to use (overrides config)',
    )
    optional.add_argument(
        '--nodes',
        type=int,
        help='Number of nodes to use (overrides config)',
    )
    # optional.add_argument(
    #     '--resume',
    #     type=Path,
    #     help='Resume training from checkpoint',
    # )
    optional.add_argument(
        '--overrides',
        nargs='*',
        metavar='KEY=VALUE',
        help='Override specific config values (format: key=value)',
    )


def handle(args):
    """Handle the train command execution."""

    # Determine output directory
    if not args.output:
        now = datetime.datetime.now()
        args.output = Path('runs').resolve() / now.strftime('%Y-%m-%d') / now.strftime('%H-%M-%S')

    args.output.mkdir(parents=True, exist_ok=True)

    # Load config
    from beast.io import load_config
    config = load_config(args.config)

    # Apply overrides
    if args.overrides:
        from beast.io import apply_config_overrides
        config = apply_config_overrides(config, args.overrides)

    # Override specific values from command line
    if args.data:
        config['data_dir'] = str(args.data)
    if args.gpus is not None:
        config['training']['num_gpus'] = args.gpus
    if args.nodes is not None:
        config['training']['num_nodes'] = args.nodes

    # Initialize model
    from beast.api.model import Model
    model = Model.from_config(config)

    # if args.resume:
    #     train_kwargs['resume_from_checkpoint'] = args.resume

    _logger.info(f'Training {type(model.model)} model')
    _logger.info(f'Output directory: {args.output}')

    # Run training
    model.train(output_dir=args.output)

    _logger.info(f'Training complete. Model saved to {args.output}')
