"""Command-line interface for beast video model pre-training package."""

import sys
from argparse import ArgumentParser

from beast.cli import formatting
from beast.cli.commands import COMMANDS


def build_parser() -> ArgumentParser:
    """Build the main argument parser with all subcommands."""

    parser = formatting.ArgumentParser(
        prog='beast',
        description='Tools for video frame extraction and neural network pretraining.',
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='Command to run',
        parser_class=formatting.SubArgumentParser,
    )

    # register all commands from the commands module
    for name, module in COMMANDS.items():
        module.register_parser(subparsers)

    return parser


def main():
    """Main CLI entry point."""

    parser = build_parser()

    # if no commands provided, display help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # parse arguments
    args = parser.parse_args()

    # get command handler
    command_handler = COMMANDS[args.command].handle

    # execute command
    command_handler(args)


if __name__ == '__main__':
    main()
