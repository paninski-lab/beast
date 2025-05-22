"""Command to run model inference on videos."""

from pathlib import Path


def register_parser(subparsers):
    """Register the predict command parser."""
    parser = subparsers.add_parser(
        "predict",
        description="Run inference using a trained model on video data.",
        usage="beast predict --model <model_dir> --input <video_path> [options]",
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Directory containing trained model",
    )
    required.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input video file or directory of videos",
    )

    # Optional arguments
    optional = parser.add_argument_group("options")
    optional.add_argument(
        "--output", "-o",
        type=Path,
        help="Directory to save prediction results (default: <model_dir>/predictions)",
    )
    optional.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    optional.add_argument(
        "--extract-layers", "-l",
        type=str,
        nargs="+",
        help="Extract features from specific layers",
    )
    optional.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations of results",
    )
    optional.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)",
    )


def handle(args):
    """Handle the predict command execution."""
    # Determine output directory
    if not args.output:
        args.output = args.model / "predictions"

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Running inference with model from: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output}")

    # Load model
    from beast.api import BeastModel
    model = BeastModel.from_dir(args.model)

    # Run prediction
    if args.input.is_file():
        # Single video inference
        results = model.predict_video(
            video_path=args.input,
            batch_size=args.batch_size,
            extract_layers=args.extract_layers,
        )

        # Save results
        from beast.utils.io import save_predictions
        save_predictions(results, args.output / f"{args.input.stem}_predictions.json")

        if args.visualize:
            from beast.visualization import visualize_predictions
            visualize_predictions(
                results,
                args.input,
                args.output / f"{args.input.stem}_visualization.mp4",
                threshold=args.threshold,
            )
    else:
        # Directory of videos
        from beast.utils.io import process_video_directory
        process_video_directory(
            model=model,
            video_dir=args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
            extract_layers=args.extract_layers,
            visualize=args.visualize,
            threshold=args.threshold,
        )

    print(f"Prediction complete. Results saved to {args.output}")
