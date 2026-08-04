import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontends"))
from frontends import desktop_bridge


class KnowledgeBaseErrorMessageTests(unittest.TestCase):
    def test_network_error_detail_is_user_facing_and_keeps_transport_diagnostics_private(self):
        error = (
            "下载 MinerU 解析结果失败：HTTPSConnectionPool(host='cdn.example', "
            "port=443): Max retries exceeded with url: /pdf/result.zip (Caused by "
            "SSLEOFError(8, 'UNEXPECTED_EOF_WHILE_READING'))"
        )

        public = desktop_bridge._kb_public_job_item({
            "name": "report.pdf",
            "status": "failed",
            "error": error,
        })

        self.assertEqual(public["errorCode"], "network_error")
        self.assertIn("网络连接失败", public["errorDetail"])
        self.assertNotIn("HTTPSConnectionPool", public["errorDetail"])
        self.assertNotIn("cdn.example", public["errorDetail"])
        self.assertNotIn("UNEXPECTED_EOF", public["errorDetail"])
        self.assertIn("cdn.example", desktop_bridge._kb_debug_error_detail(error))
