import json
import unittest
from types import SimpleNamespace

from agent_loop import BaseHandler, StepOutcome, agent_runner_loop
from ga import GenericAgentHandler


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


class _ContinuingToolHandler(_ToolHandler):
    def do_demo(self, args, response):
        return StepOutcome({"answer": "not final"}, next_prompt="continue")


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
    @staticmethod
    def _run_generator(generator):
        chunks = []
        try:
            while True:
                chunks.append(next(generator))
        except StopIteration as stopped:
            return chunks, stopped.value

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

    def test_answer_followed_by_summary_is_not_retried(self):
        parent = SimpleNamespace(knowledge_scope={"mode": "none"})
        handler = GenericAgentHandler(parent)
        response = SimpleNamespace(
            content="这是正式回答。\n<summary>已回答</summary>",
            thinking="",
        )

        _chunks, outcome = self._run_generator(handler.do_no_tool({}, response))

        self.assertIsNone(outcome.next_prompt)
        self.assertFalse(outcome.should_exit)

    def test_summary_only_retries_then_emits_visible_failure(self):
        parent = SimpleNamespace(knowledge_scope={"mode": "none"})
        handler = GenericAgentHandler(parent)
        response = SimpleNamespace(content="<summary>尚未正式回答</summary>", thinking="")

        outcomes = []
        chunks = []
        for _ in range(3):
            emitted, outcome = self._run_generator(handler.do_no_tool({}, response))
            chunks.extend(emitted)
            outcomes.append(outcome)

        self.assertTrue(outcomes[-1].should_exit)
        self.assertIn("用户可见答复", "".join(chunks))

    def test_valid_response_resets_incomplete_retry_counter(self):
        parent = SimpleNamespace(knowledge_scope={"mode": "none"})
        handler = GenericAgentHandler(parent)
        incomplete = SimpleNamespace(content="<summary>未完成</summary>", thinking="")
        complete = SimpleNamespace(content="正式回答", thinking="")

        self._run_generator(handler.do_no_tool({}, incomplete))
        self._run_generator(handler.do_no_tool({}, complete))
        _chunks, outcome = self._run_generator(handler.do_no_tool({}, incomplete))

        self.assertFalse(outcome.should_exit)

    def test_max_turns_emits_user_visible_failure(self):
        chunks = list(agent_runner_loop(
            _Client(), "system", "question", _ContinuingToolHandler(), [],
            max_turns=1, verbose=True,
        ))

        text = "".join(chunk for chunk in chunks if isinstance(chunk, str))
        self.assertIn("允许的轮次内形成完整答复", text)


if __name__ == "__main__":
    unittest.main()
