"""Tests for the beast.logging module."""

from unittest.mock import MagicMock

from beast.logging import log_step


class TestLogStep:
    """Test the log_step function."""

    def test_plain_message_printed_with_timestamp(self, capsys) -> None:
        log_step('hello world')
        captured = capsys.readouterr()
        assert 'hello world' in captured.out
        assert 'INFO' not in captured.out
        assert 'DEBUG' not in captured.out
        assert 'ERROR' not in captured.out

    def test_info_level_prints_info_prefix(self, capsys) -> None:
        log_step('info message', level='info')
        captured = capsys.readouterr()
        assert 'INFO: info message' in captured.out

    def test_debug_level_prints_debug_prefix(self, capsys) -> None:
        log_step('debug message', level='debug')
        captured = capsys.readouterr()
        assert 'DEBUG: debug message' in captured.out

    def test_error_level_prints_error_prefix(self, capsys) -> None:
        log_step('error message', level='error')
        captured = capsys.readouterr()
        assert 'ERROR: error message' in captured.out

    def test_unknown_level_falls_through_to_plain(self, capsys) -> None:
        log_step('plain', level='warning')
        captured = capsys.readouterr()
        assert 'plain' in captured.out
        assert 'WARNING' not in captured.out

    def test_info_with_logger_delegates_to_logger_info(self) -> None:
        mock_logger = MagicMock()
        log_step('msg', level='info', logger=mock_logger)
        mock_logger.info.assert_called_once_with('msg')
        mock_logger.debug.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_debug_with_logger_delegates_to_logger_debug(self) -> None:
        mock_logger = MagicMock()
        log_step('msg', level='debug', logger=mock_logger)
        mock_logger.debug.assert_called_once_with('msg')
        mock_logger.info.assert_not_called()

    def test_error_with_logger_delegates_to_logger_error(self) -> None:
        mock_logger = MagicMock()
        log_step('msg', level='error', logger=mock_logger)
        mock_logger.error.assert_called_once_with('msg')
        mock_logger.info.assert_not_called()

    def test_with_logger_does_not_print(self, capsys) -> None:
        mock_logger = MagicMock()
        log_step('msg', level='info', logger=mock_logger)
        captured = capsys.readouterr()
        assert captured.out == ''
