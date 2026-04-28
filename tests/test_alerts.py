"""Unit tests for alerts.py — send_alert_with_attachments."""

import email
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eureka"))

from alerts import send_alert_with_attachments


def _make_cfg(enabled: bool = True) -> types.SimpleNamespace:
    alerts = types.SimpleNamespace(
        enabled=enabled,
        gmail_user="sender@gmail.com",
        gmail_password="secret",
        recipient="recipient@example.com",
    )
    return types.SimpleNamespace(alerts=alerts)


class TestSendAlertWithAttachments(unittest.TestCase):
    def _write_temp_gif(self, directory: str, name: str = "test.gif") -> str:
        path = os.path.join(directory, name)
        with open(path, "wb") as f:
            f.write(b"GIF89a fake gif content")
        return path

    @patch("smtplib.SMTP_SSL")
    def test_sendmail_called_with_attachment(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with tempfile.TemporaryDirectory() as d:
            gif = self._write_temp_gif(d)
            cfg = _make_cfg(enabled=True)
            send_alert_with_attachments(cfg, "Test subject", "Test body", [gif])
            mock_server.sendmail.assert_called_once()

    @patch("smtplib.SMTP_SSL")
    def test_message_contains_attachment_part(self, mock_smtp_cls):
        captured = {}

        def fake_sendmail(from_addr, to_addr, msg_string):
            captured["msg"] = email.message_from_string(msg_string)

        mock_server = MagicMock()
        mock_server.sendmail.side_effect = fake_sendmail
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with tempfile.TemporaryDirectory() as d:
            gif = self._write_temp_gif(d, "episode_0.gif")
            cfg = _make_cfg(enabled=True)
            send_alert_with_attachments(cfg, "subj", "body", [gif])

        msg = captured["msg"]
        filenames = [
            part.get_filename()
            for part in msg.walk()
            if part.get_filename() is not None
        ]
        self.assertIn("episode_0.gif", filenames)

    @patch("smtplib.SMTP_SSL")
    def test_no_send_when_disabled(self, mock_smtp_cls):
        with tempfile.TemporaryDirectory() as d:
            gif = self._write_temp_gif(d)
            cfg = _make_cfg(enabled=False)
            send_alert_with_attachments(cfg, "subj", "body", [gif])
            mock_smtp_cls.assert_not_called()

    @patch("smtplib.SMTP_SSL")
    def test_empty_attachments_still_sends(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        cfg = _make_cfg(enabled=True)
        send_alert_with_attachments(cfg, "subj", "body", [])
        mock_server.sendmail.assert_called_once()

    @patch("smtplib.SMTP_SSL")
    def test_multiple_attachments(self, mock_smtp_cls):
        captured = {}

        def fake_sendmail(from_addr, to_addr, msg_string):
            captured["msg"] = email.message_from_string(msg_string)

        mock_server = MagicMock()
        mock_server.sendmail.side_effect = fake_sendmail
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with tempfile.TemporaryDirectory() as d:
            gifs = [self._write_temp_gif(d, f"ep_{i}.gif") for i in range(3)]
            cfg = _make_cfg(enabled=True)
            send_alert_with_attachments(cfg, "subj", "body", gifs)

        msg = captured["msg"]
        filenames = [
            part.get_filename()
            for part in msg.walk()
            if part.get_filename() is not None
        ]
        self.assertEqual(len(filenames), 3)


if __name__ == "__main__":
    unittest.main()
