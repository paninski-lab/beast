"""Tests for CLI formatting classes."""

import argparse

import pytest

from beast.cli.formatting import ArgumentParser, HelpFormatter, SubArgumentParser


class TestArgumentParser:
    """Test the ArgumentParser class."""

    def test_is_sub_parser_false(self) -> None:
        # Act
        parser = ArgumentParser()
        # Assert
        assert parser.is_sub_parser is False

    def test_error_exits_with_code_2(self) -> None:
        # Arrange
        parser = ArgumentParser()
        # Act / Assert
        with pytest.raises(SystemExit) as exc_info:
            parser.error('something went wrong')
        assert exc_info.value.code == 2

    def test_print_help_includes_banner(self, capsys) -> None:
        # Arrange
        parser = ArgumentParser(prog='beast')
        # Act
        parser.print_help()
        # Assert
        captured = capsys.readouterr()
        assert 'BEAST' in captured.out


class TestSubArgumentParser:
    """Test the SubArgumentParser class."""

    def test_is_sub_parser_true(self) -> None:
        # Act
        parser = SubArgumentParser()
        # Assert
        assert parser.is_sub_parser is True

    def test_print_help_no_banner(self, capsys) -> None:
        # Arrange
        parser = SubArgumentParser(prog='beast extract')
        # Act
        parser.print_help()
        # Assert
        captured = capsys.readouterr()
        assert 'BEAST' not in captured.out


class TestHelpFormatter:
    """Test the HelpFormatter class."""

    def test_split_lines_preserves_newlines(self) -> None:
        # Arrange
        formatter = HelpFormatter('prog')
        # Act
        lines = formatter._split_lines('line one\nline two', 80)
        # Assert
        assert lines[0] == 'line one'
        assert lines[1] == 'line two'

    def test_split_lines_empty_paragraph_becomes_blank_line(self) -> None:
        # Arrange
        formatter = HelpFormatter('prog')
        # Act
        lines = formatter._split_lines('a\n\nb', 80)
        # Assert
        assert '' in lines

    def test_format_action_adds_trailing_newline(self) -> None:
        # Arrange
        parser = argparse.ArgumentParser(formatter_class=HelpFormatter)
        action = parser.add_argument('--foo', help='foo help')
        formatter = parser._get_formatter()
        # Act
        result = formatter._format_action(action)
        # Assert
        assert result.endswith('\n\n')
