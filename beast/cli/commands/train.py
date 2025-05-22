"""Command to train a model."""

import datetime
from pathlib import Path

from beast.cli.types import config_file, output_dir


def register_parser(subparsers):
    """Register the train command parser."""
    parser = subparsers.add_parser(
        "train",
        description="Train a neural network model on video frame data.",
        usage="beast train --config <config_path> [options]",
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--config", "-c",
        type=config_file,
        required=True,
        help="Path to model configuration file (YAML or JSON)",
    )

    # Optional arguments
    optional = parser.add_argument_group("options")
    optional.add_argument(
        "--output", "-o",
        type=output_dir,
        help="Directory to save model outputs (default: ./runs/YYYY-MM-DD/HH-MM-SS)",
    )
    optional.add_argument(
        "--data", "-d",
        type=Path,
        help="Override data directory specified in config",
    )
    optional.add_argument(
        "--gpus",
        type=int,
        help="Number of GPUs to use (overrides config)",
    )
    # optional.add_argument(
    #     "--resume",
    #     type=Path,
    #     help="Resume training from checkpoint",
    # )
    optional.add_argument(
        "--overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override specific config values (format: key=value)",
    )


def handle(args):
    """Handle the train command execution."""
    # Determine output directory
    if not args.output:
        now = datetime.datetime.now()
        args.output = Path("runs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Training model with config: {args.config}")
    print(f"Output directory: {args.output}")

    # Load config
    from beast.utils.config import load_config
    config = load_config(args.config)

    # Apply overrides
    if args.overrides:
        from beast.utils.config import apply_overrides
        config = apply_overrides(config, args.overrides)

    # Override specific values from command line
    if args.data:
        config["data_dir"] = str(args.data)
    if args.gpus is not None:
        config["trainer"]["gpus"] = args.gpus
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name

    # Initialize model
    from beast.api.model import Model
    model = Model.from_config(config)

    # Train model
    train_kwargs = {
        "output_dir": args.output,
    }

    # if args.resume:
    #     train_kwargs["resume_from_checkpoint"] = args.resume

    # Run training
    model.train(**train_kwargs)

    print(f"Training complete. Model saved to {args.output}")
