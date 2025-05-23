"""Custom formatting for CLI help and error messages."""

import argparse
import sys
import textwrap


class ArgumentParser(argparse.ArgumentParser):
    """Enhanced argument parser with better formatting."""

    def __init__(self, **kwargs):
        super().__init__(
            formatter_class=HelpFormatter,
            **kwargs
        )
        self.is_sub_parser = False

    def print_help(self, file=None, **kwargs):
        """Print help message with optional welcome text."""
        if not self.is_sub_parser:
            print("\nBEAST - Self-supervised pretraing of neural networks on video data\n")
        super().print_help(file=file)

    def error(self, message):
        """Print error message with colorized output."""
        red = '\033[91m'
        reset = '\033[0m'
        sys.stderr.write(f'{red}Error: {message}{reset}\n\n')
        self.print_help(file=sys.stderr)
        sys.exit(2)


class SubArgumentParser(ArgumentParser):
    """Argument parser for subcommands."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_sub_parser = True


class HelpFormatter(argparse.HelpFormatter):
    """Custom formatter for better help text readability."""

    def _split_lines(self, text, width):
        """Preserve newlines and handle long text better."""
        paragraphs = text.splitlines()
        lines = []

        for p in paragraphs:
            p_lines = textwrap.wrap(
                p, width,
                break_long_words=False,
                break_on_hyphens=False
            )
            if not p_lines:
                p_lines = ['']
            lines.extend(p_lines)

        return lines

    def _fill_text(self, text, width, indent):
        """Improved text filling with indentation."""
        return '\n'.join(
            indent + line for line in self._split_lines(text, width - len(indent))
        )

    def _format_action(self, action):
        """Add spacing between arguments for readability."""
        result = super()._format_action(action)
        return result + '\n'
