import json
import unittest
from types import SimpleNamespace

from agent_loop import BaseHandler, StepOutcome, agent_runner_loop


class _ToolHandler(BaseHandler):
    def __init__(self):
        self.parent = SimpleNamespace(task_dir=None)
        self._done_hooks = []
        self.current_turn = 0
        self.max_turns = 1

    def do_demo(self, args, response):
        return StepOutcome({"answer": "real tool payload", "count": 2}, next_prompt=None)


class _StreamingToolHandler(_ToolHandler):
    def do_demo(self, args, response):
        yield "real streamed payload\n"
        return StepOutcome("real streamed payload", next_prompt=None)


class _NoToolClient:
    last_tools = ""

    def chat(self, messages, tools):
        response = SimpleNamespace(content="answer", thinking="", tool_calls=[])
        yield "answer"
        return response


class _Client:
    last_tools = ""

    def chat(self, messages, tools):
        response = SimpleNamespace(
            content="",
            thinking="",
            tool_calls=[SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="demo", arguments=json.dumps({"value": 1})),
            )],
        )
        if False:
            yield ""
        return response


class AgentLoopResultTests(unittest.TestCase):
    def test_verbose_tool_stream_contains_returned_payload(self):
        chunks = list(agent_runner_loop(
            _Client(),
            "system",
            "question",
            _ToolHandler(),
            [{"type": "function", "function": {"name": "demo"}}],
            max_turns=1,
            verbose=True,
        ))

        text = "".join(chunk for chunk in chunks if isinstance(chunk, str))
        self.assertIn('"answer": "real tool payload"', text)
        self.assertIn('"count": 2', text)

    def test_streamed_tool_payload_is_not_appended_twice(self):
        chunks = list(agent_runner_loop(
            _Client(), "system", "question", _StreamingToolHandler(), [],
            max_turns=1, verbose=True,
        ))

        text = "".join(chunk for chunk in chunks if isinstance(chunk, str))
        self.assertEqual(text.count("real streamed payload"), 1)

    def test_final_answer_has_no_orphan_tool_result_fence(self):
        chunks = list(agent_runner_loop(
            _NoToolClient(), "system", "question", _ToolHandler(), [],
            max_turns=1, verbose=True,
        ))

        text = "".join(chunk for chunk in chunks if isinstance(chunk, str))
        self.assertEqual(text.count("`````"), 0)


if __name__ == "__main__":
    unittest.main()
