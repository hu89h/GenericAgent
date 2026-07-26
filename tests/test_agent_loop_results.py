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


if __name__ == "__main__":
    unittest.main()
