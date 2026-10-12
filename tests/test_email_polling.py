"""Regression tests for Facebook verification mail polling."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "faceautovn"))
import main


class EmailPollingTests(unittest.TestCase):
    def test_rate_limit_then_facebook_subject(self):
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "12"
        error = requests.HTTPError(response=response)
        inbox = {
            "data": {
                "items": [{"sender_name": "Facebook", "subject": "66981 là mã xác nhận Facebook của bạn"}]
            }
        }
        with patch.object(main, "read_email", side_effect=[error, inbox]) as read, \
                patch.object(main.time, "sleep") as sleep:
            self.assertEqual(main.read_email_by_id({"id": 123}), "66981")
        self.assertEqual(read.call_count, 2)
        sleep.assert_called_once_with(12)

    def test_old_subject_format(self):
        inbox = {
            "data": {
                "items": [{"sender_name": "Facebook", "subject": "123456 là mã xác nhận của bạn"}]
            }
        }
        with patch.object(main, "read_email", return_value=inbox):
            self.assertEqual(main.read_email_by_id({"id": 123}), "123456")


if __name__ == "__main__":
    unittest.main()
