import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import llmcore
import multimodal
from agentmain import GenericAgent
from ga import GenericAgentHandler


def _finish(generator):
    while True:
        try:
            next(generator)
        except StopIteration as stopped:
            return stopped.value


class MultimodalNormalizationTests(unittest.TestCase):
    def setUp(self):
        multimodal.clear_image_cache()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        multimodal.clear_image_cache()
        self.temp.cleanup()

    def _save(self, name, image, fmt=None, **kwargs):
        path = self.root / name
        image.save(path, format=fmt, **kwargs)
        return path

    def test_supported_formats_are_normalized_and_alpha_is_flattened(self):
        sources = [
            self._save("sample.png", Image.new("RGBA", (8, 6), (255, 0, 0, 80))),
            self._save("sample.jpg", Image.new("RGB", (8, 6), "red"), "JPEG"),
            self._save("sample.webp", Image.new("RGB", (8, 6), "red"), "WEBP"),
            self._save("sample.gif", Image.new("P", (8, 6)), "GIF"),
            self._save("sample.bmp", Image.new("RGB", (8, 6), "red"), "BMP"),
            self._save("sample.tiff", Image.new("RGB", (8, 6), "red"), "TIFF"),
            self._save("sample.ico", Image.new("RGBA", (32, 32), "red"), "ICO"),
        ]

        prepared = [multimodal.prepare_image(path) for path in sources]

        self.assertTrue(all(item.width > 0 and item.height > 0 for item in prepared))
        self.assertTrue(all(item.media_type in {"image/png", "image/jpeg"} for item in prepared))
        png = Image.open(io.BytesIO(__import__("base64").b64decode(prepared[0].data)))
        self.assertNotIn("A", png.mode)

    def test_exif_orientation_and_pixel_resize(self):
        path = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (40, 20), "blue").save(path, exif=exif)

        oriented = multimodal.prepare_image(path)
        self.assertEqual((oriented.width, oriented.height), (20, 40))

        large = self._save("large.png", Image.new("RGB", (1000, 1000), "green"))
        resized = multimodal.prepare_image(large, max_pixels=10_000)
        self.assertLessEqual(resized.width * resized.height, 10_000)

    def test_limits_and_corrupt_images_have_structured_errors(self):
        path = self._save("small.png", Image.new("RGB", (10, 10), "red"))
        with self.assertRaises(multimodal.ImageContentError) as source_error:
            multimodal.prepare_image(path, max_source_bytes=1)
        self.assertEqual(source_error.exception.code, "image_source_too_large")

        with self.assertRaises(multimodal.ImageContentError) as pixel_error:
            multimodal.prepare_image(path, max_decoded_pixels=20)
        self.assertEqual(pixel_error.exception.code, "image_dimensions_invalid")

        with self.assertRaises(multimodal.ImageContentError) as encoded_error:
            multimodal.prepare_image(path, max_encoded_bytes=1)
        self.assertEqual(encoded_error.exception.code, "image_encoded_too_large")

        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with self.assertRaises(multimodal.ImageContentError) as decode_error:
            multimodal.prepare_image(corrupt)
        self.assertEqual(decode_error.exception.code, "image_decode_failed")

    def test_cache_reuses_normalized_result_for_the_same_contract(self):
        path = self._save("cached.png", Image.new("RGB", (20, 20), "red"))
        with mock.patch.object(multimodal, "_encode_image", wraps=multimodal._encode_image) as encode:
            first = multimodal.prepare_image(path)
            second = multimodal.prepare_image(path)
            third = multimodal.prepare_image(path, max_pixels=100)

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(encode.call_count, 2)

    def test_cache_evicts_least_recently_used_images_at_capacity(self):
        first = self._save("first.png", Image.new("RGB", (20, 20), "red"))
        second = self._save("second.png", Image.new("RGB", (20, 20), "blue"))
        with mock.patch.object(multimodal, "_CACHE_MAX_BYTES", 100), \
             mock.patch.object(multimodal, "_encode_image", wraps=multimodal._encode_image) as encode:
            multimodal.prepare_image(first)
            multimodal.prepare_image(second)
            multimodal.prepare_image(first)
        self.assertEqual(encode.call_count, 3)

    def test_protocol_materialization_does_not_mutate_history(self):
        path = self._save("protocol.png", Image.new("RGB", (3, 2), "blue"))
        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                multimodal.image_path_block(path, "figure.png"),
            ],
        }]
        original = copy.deepcopy(history)

        anthropic = multimodal.materialize_anthropic_messages(history)
        chat = llmcore._msgs_claude2oai(history)
        responses = llmcore._to_responses_input(chat)

        self.assertEqual(history, original)
        self.assertEqual(anthropic[0]["content"][1]["type"], "image")
        self.assertEqual(anthropic[0]["content"][1]["source"]["type"], "base64")
        self.assertEqual(chat[0]["content"][1]["type"], "image_url")
        self.assertEqual(responses[0]["content"][1]["type"], "input_image")
        self.assertNotIn("base64", json.dumps(history))
        self.assertNotIn("data:image", json.dumps(history))

    def test_logging_redacts_paths_and_encoded_image_data(self):
        path = self._save("secret-name.png", Image.new("RGB", (3, 2), "blue"))
        value = {
            "path": multimodal.image_path_block(path),
            "url": '{"image":"data:image/png;base64,AAAA"}',
        }
        safe = multimodal.safe_log_value(value)
        rendered = json.dumps(safe)
        self.assertNotIn(str(self.root), rendered)
        self.assertNotIn("AAAA", rendered)
        self.assertIn("secret-name.png", rendered)

    def test_missing_persisted_image_becomes_explicit_text(self):
        history = [{
            "role": "user",
            "content": [{"type": "image_path", "path": str(self.root / "gone.png"), "name": "gone.png"}],
        }]
        restored = multimodal.restore_image_references(history)
        self.assertEqual(restored[0]["content"][0]["type"], "text")
        self.assertIn("no longer available", restored[0]["content"][0]["text"])


class AgentImageRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.image = Path(self.temp.name) / "image.png"
        Image.new("RGB", (4, 4), "red").save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_tool_client_retains_image_blocks(self):
        captured = []

        class Backend:
            name = "vision"
            model = "test"
            history = []
            supports_vision = True
            system = ""
            tools = None

            def ask(self, message):
                captured.append(message)
                if False:
                    yield None
                return SimpleNamespace(raw="ok", tool_calls=[])

        client = llmcore.NativeToolClient(Backend())
        block = multimodal.image_path_block(self.image)
        _finish(client.chat([{"role": "user", "content": [{"type": "text", "text": "look"}, block]}]))
        self.assertEqual(captured[0]["content"][1], block)

    def test_generic_agent_rejects_image_before_queueing_for_text_client(self):
        agent = GenericAgent.__new__(GenericAgent)
        agent.llmclient = SimpleNamespace(supports_image_blocks=False)
        agent.task_queue = __import__("queue").Queue()
        with self.assertRaises(multimodal.ImageContentError) as error:
            agent.put_task("look", images=[str(self.image)])
        self.assertEqual(error.exception.code, "vision_model_required")
        self.assertTrue(agent.task_queue.empty())

    def test_initial_content_removes_ui_placeholder_and_preserves_image_order(self):
        first = multimodal.image_path_block(self.image, "first.png")
        second = multimodal.image_path_block(self.image, "second.png")
        content = GenericAgent._initial_user_content(
            "compare [Image #9] then [Image #2]",
            [first, second],
        )
        self.assertEqual(content[0]["text"], "compare then")
        self.assertEqual([block.get("name") for block in content if block.get("type") == "image_path"], [
            "first.png", "second.png",
        ])
        self.assertNotIn("[Image #", json.dumps(content))

    def test_mixin_routes_image_only_to_visual_members(self):
        used = []

        class Node:
            def __init__(self, name, vision):
                self.name = name
                self.supports_vision = vision
                self.system = ""
                self.tools = None

            def raw_ask(self, _messages):
                used.append(self.name)
                yield self.name
                return [{"type": "text", "text": self.name}]

        mixin = llmcore.MixinSession.__new__(llmcore.MixinSession)
        object.__setattr__(mixin, "_sessions", [Node("text", False), Node("vision", True)])
        object.__setattr__(mixin, "_native", True)
        object.__setattr__(mixin, "_cur_idx", 0)
        object.__setattr__(mixin, "_switched_at", 0.0)
        object.__setattr__(mixin, "_spring_sec", 300)
        object.__setattr__(mixin, "_retries", 0)
        object.__setattr__(mixin, "_base_delay", 0)
        object.__setattr__(mixin, "system", "")
        object.__setattr__(mixin, "tools", None)
        messages = [{"role": "user", "content": [multimodal.image_path_block(self.image)]}]

        _finish(mixin.raw_ask(messages))
        self.assertEqual(used, ["vision"])

    def test_file_read_queues_image_for_the_next_turn(self):
        parent = SimpleNamespace(llmclient=SimpleNamespace(supports_image_blocks=True))
        handler = GenericAgentHandler(parent, cwd=self.temp.name)
        outcome = _finish(handler.do_file_read({"path": str(self.image)}, None))
        blocks = handler.take_pending_inline_blocks()

        self.assertEqual(outcome.data["attach_status"], "attached")
        self.assertEqual(blocks[-1]["type"], "image_path")
        self.assertEqual(blocks[-1]["path"], str(self.image.resolve()))


if __name__ == "__main__":
    unittest.main()
