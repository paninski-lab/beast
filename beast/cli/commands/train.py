"""Command to train a model."""

import datetime
import logging
from pathlib import Path

from beast import log_step
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

    log_step("Starting train command handler", level='info', logger=_logger)

    # Determine output directory
    log_step("Determining output directory", level='info', logger=_logger)
    if not args.output:
        now = datetime.datetime.now()
        args.output = Path('runs').resolve() / now.strftime('%Y-%m-%d') / now.strftime('%H-%M-%S')

    args.output.mkdir(parents=True, exist_ok=True)
    log_step(f"Output directory: {args.output}", level='info', logger=_logger)

    # Set up logging to the model directory
    log_step("Setting up model logging", level='info', logger=_logger)
    model_log_handler = _setup_model_logging(args.output)
    log_step("Model logging set up", level='info', logger=_logger)

    # try:

    # Load config
    log_step(f"Loading config from: {args.config}", level='info', logger=_logger)
    from beast.io import load_config
    config = load_config(args.config)
    log_step("Config loaded", level='info', logger=_logger)

    # Apply overrides
    if args.overrides:
        log_step("Applying config overrides", level='info', logger=_logger)
        from beast.io import apply_config_overrides
        config = apply_config_overrides(config, args.overrides)
        log_step("Config overrides applied", level='info', logger=_logger)

    # Override specific values from command line
    log_step("Applying command line overrides", level='info', logger=_logger)
    if args.data:
        config['data']['data_dir'] = str(args.data)
        log_step(f"Data directory overridden to: {args.data}", level='info', logger=_logger)
    if args.gpus is not None:
        config['training']['num_gpus'] = args.gpus
        log_step(f"Number of GPUs overridden to: {args.gpus}", level='info', logger=_logger)
    if args.nodes is not None:
        config['training']['num_nodes'] = args.nodes
        log_step(f"Number of nodes overridden to: {args.nodes}", level='info', logger=_logger)

    # Check for unsupported --checkpoint argument
    if hasattr(args, 'checkpoint') and args.checkpoint:
        log_step(
            f"WARNING: --checkpoint argument provided but not supported: {args.checkpoint}", level='info', logger=_logger)
        log_step("Checkpoint resuming is not currently implemented in the CLI",
                 level='info', logger=_logger)

    # Initialize model
    log_step("Initializing model from config", level='info', logger=_logger)
    from beast.api.model import Model
    model = Model.from_config(config)
    log_step("Model initialized", level='info', logger=_logger)

    # if args.resume:
    #     train_kwargs['resume_from_checkpoint'] = args.resume
    log_step(f'Training {type(model.model)} model', level='info', logger=_logger)
    log_step(f'Data directory: {args.data}', level='info', logger=_logger)
    log_step(f'Output directory: {args.output}', level='info', logger=_logger)

    # Run training
    log_step("About to call model.train()", level='info', logger=_logger)
    model.train(output_dir=args.output)
    log_step("model.train() completed", level='info', logger=_logger)
    log_step(f'Training complete. Model saved to {args.output}', level='info', logger=_logger)

    # except Exception as e:

    #     _logger.error(e)

    # finally:

    #     # Clean up the handler when done
    #     root_logger = logging.getLogger()
    #     root_logger.removeHandler(model_log_handler)
    #     model_log_handler.close()


def _setup_model_logging(output_dir: Path):
    """Set up additional logging to the model directory and remove original file handler."""

    # Create log file path
    log_file = output_dir / 'training.log'

    # Get the root logger
    root_logger = logging.getLogger()

    # Create a new file handler for the model directory
    model_handler = logging.FileHandler(log_file)
    model_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s  %(name)s : %(message)s'
    )
    model_handler.setFormatter(formatter)

    # Add the new handler to the root logger
    root_logger.addHandler(model_handler)

    return model_handler
