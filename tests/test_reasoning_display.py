import json
import unittest

import llmcore


def drain(generator):
    chunks = []
    try:
        while True:
            chunks.append(next(generator))
    except StopIteration as stopped:
        return "".join(chunks), stopped.value


def sse(*events):
    return [f"data: {json.dumps(event)}" for event in events]


class ReasoningDisplayTests(unittest.TestCase):
    def test_claude_json_exposes_thinking_only_in_display_stream(self):
        shown, blocks = drain(llmcore._parse_claude_json({
            "content": [
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": "answer"},
            ],
        }))
        self.assertEqual(shown, "<thinking>reason</thinking>\n\nanswer")
        self.assertEqual([block["type"] for block in blocks], ["thinking", "text"])

    def test_claude_stream_wraps_thinking_deltas(self):
        shown, blocks = drain(llmcore._parse_claude_sse(sse(
            {"type": "message_start", "message": {"usage": {}}},
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "reason"}},
            {"type": "content_block_stop"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "answer"}},
            {"type": "content_block_stop"},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
            {"type": "message_stop"},
        )))
        self.assertEqual(shown, "<thinking>reason</thinking>\n\nanswer")
        self.assertEqual([block["type"] for block in blocks], ["thinking", "text"])

    def test_openai_chat_stream_separates_reasoning_from_answer(self):
        shown, blocks = drain(llmcore._parse_openai_sse(sse(
            {"choices": [{"delta": {"reasoning_content": "reason"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
        )))
        self.assertEqual(shown, "<thinking>reason</thinking>\n\nanswer")
        self.assertEqual([block["type"] for block in blocks], ["thinking", "text"])

    def test_openai_json_separates_reasoning_from_answer(self):
        shown, blocks = drain(llmcore._parse_openai_json({
            "choices": [{"message": {"reasoning_content": "reason", "content": "answer"}}],
        }))
        self.assertEqual(shown, "<thinking>reason</thinking>\n\nanswer")
        self.assertEqual([block["type"] for block in blocks], ["thinking", "text"])

    def test_responses_stream_exposes_reasoning_summary(self):
        shown, blocks = drain(llmcore._parse_openai_sse(sse(
            {"type": "response.reasoning_summary_text.delta", "delta": "reason"},
            {"type": "response.reasoning_summary_text.done", "text": "reason"},
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed", "response": {"usage": {}}},
        ), "responses"))
        self.assertEqual(shown, "<thinking>reason</thinking>\n\nanswer")
        self.assertEqual([block["type"] for block in blocks], ["thinking", "text"])


if __name__ == "__main__":
    unittest.main()
