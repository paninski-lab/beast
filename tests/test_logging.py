"""Tests for the beast.logging module."""

import logging
from unittest.mock import MagicMock

from beast.logging import log_step


class TestLogStep:
    """Test the log_step function."""

    def test_plain_message_logs_at_info(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger='beast'):
            log_step('hello world')
        assert 'hello world' in caplog.text

    def test_info_level_logs_at_info(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger='beast'):
            log_step('info message', level='info')
        assert 'info message' in caplog.text
        assert caplog.records[-1].levelno == logging.INFO

    def test_debug_level_logs_at_debug(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger='beast'):
            log_step('debug message', level='debug')
        assert 'debug message' in caplog.text
        assert caplog.records[-1].levelno == logging.DEBUG

    def test_error_level_logs_at_error(self, caplog) -> None:
        with caplog.at_level(logging.ERROR, logger='beast'):
            log_step('error message', level='error')
        assert 'error message' in caplog.text
        assert caplog.records[-1].levelno == logging.ERROR

    def test_unknown_level_falls_through_to_info(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger='beast'):
            log_step('plain', level='warning')
        assert 'plain' in caplog.text
        assert caplog.records[-1].levelno == logging.INFO

    def test_flush_param_accepted(self) -> None:
        # flush is a no-op but must not raise
        log_step('msg', flush=False)

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

    def test_with_logger_does_not_use_default_logger(self, caplog) -> None:
        mock_logger = MagicMock()
        with caplog.at_level(logging.DEBUG, logger='beast'):
            log_step('msg', level='info', logger=mock_logger)
        assert 'msg' not in caplog.text
