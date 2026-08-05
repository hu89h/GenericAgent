import json
import unittest
from types import SimpleNamespace

from agent_loop import BaseHandler, StepOutcome, agent_runner_loop
from ga import GenericAgentHandler
from llmcore import _parse_text_tool_calls


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


class _FinalizingHandler(GenericAgentHandler):
    def __init__(self):
        parent = SimpleNamespace(
            task_dir=None, knowledge_scope={"mode": "none"},
            extrakeyinfo=None, intervene=None, _turn_end_hooks={},
            verbose=False,
        )
        super().__init__(parent)

    def do_demo(self, args, response):
        return StepOutcome({"answer": "not final"}, next_prompt="continue")


class _NoToolClient:
    last_tools = ""

    def chat(self, messages, tools):
        response = SimpleNamespace(content="answer", thinking="", tool_calls=[])
        yield "answer"
        return response


class _SequenceClient:
    last_tools = ""

    def __init__(self, responses):
        self.responses = list(responses)
        self.tool_sets = []

    def chat(self, messages, tools):
        self.tool_sets.append(list(tools or []))
        response = self.responses.pop(0)
        if response.content:
            yield response.content
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

        _first_chunks, first = self._run_generator(handler.do_no_tool({}, response))
        _second_chunks, second = self._run_generator(handler.do_no_tool({}, response))
        handler._finalization_only_active = True
        chunks, final = self._run_generator(handler.do_no_tool({}, response))

        self.assertFalse(first.finalize_only)
        self.assertTrue(second.finalize_only)
        self.assertTrue(final.should_exit)
        self.assertIn("用户可见答复", "".join(chunks))

    def test_tool_turn_resets_consecutive_incomplete_counter(self):
        parent = SimpleNamespace(
            knowledge_scope={"mode": "none"}, task_dir=None,
            extrakeyinfo=None, intervene=None, _turn_end_hooks={},
        )
        handler = GenericAgentHandler(parent)
        incomplete = SimpleNamespace(content="<summary>未完成</summary>", thinking="")

        self._run_generator(handler.do_no_tool({}, incomplete))
        handler.turn_end_callback(
            SimpleNamespace(content="<summary>调用工具</summary>"),
            [{"tool_name": "demo", "args": {}}], [], 2, "continue", {},
        )
        _chunks, outcome = self._run_generator(handler.do_no_tool({}, incomplete))

        self.assertFalse(outcome.should_exit)
        self.assertFalse(outcome.finalize_only)

    def test_second_consecutive_incomplete_requests_toolless_finalization(self):
        parent = SimpleNamespace(knowledge_scope={"mode": "none"})
        handler = GenericAgentHandler(parent)
        response = SimpleNamespace(content="<summary>未完成</summary>", thinking="")

        _first_chunks, first = self._run_generator(handler.do_no_tool({}, response))
        _second_chunks, second = self._run_generator(handler.do_no_tool({}, response))

        self.assertFalse(first.finalize_only)
        self.assertTrue(second.finalize_only)
        self.assertFalse(second.should_exit)

    def test_consecutive_incomplete_responses_run_one_toolless_final_turn(self):
        incomplete = SimpleNamespace(
            content="<summary>仍在整理</summary>", thinking="", tool_calls=[],
        )
        final = SimpleNamespace(content="最终回答", thinking="", tool_calls=[])
        client = _SequenceClient([incomplete, incomplete, final])

        chunks = list(agent_runner_loop(
            client, "system", "question", _FinalizingHandler(),
            [{"type": "function", "function": {"name": "demo"}}],
            max_turns=5, verbose=True,
        ))

        text = "".join(chunk for chunk in chunks if isinstance(chunk, str))
        self.assertIn("最终回答", text)
        self.assertEqual(len(client.tool_sets), 3)
        self.assertTrue(client.tool_sets[0])
        self.assertTrue(client.tool_sets[1])
        self.assertEqual(client.tool_sets[2], [])

    def test_valid_response_resets_incomplete_retry_counter(self):
        parent = SimpleNamespace(knowledge_scope={"mode": "none"})
        handler = GenericAgentHandler(parent)
        incomplete = SimpleNamespace(content="<summary>未完成</summary>", thinking="")
        complete = SimpleNamespace(content="正式回答", thinking="")

        self._run_generator(handler.do_no_tool({}, incomplete))
        self._run_generator(handler.do_no_tool({}, complete))
        _chunks, outcome = self._run_generator(handler.do_no_tool({}, incomplete))

        self.assertFalse(outcome.should_exit)

    def test_max_turns_gets_one_toolless_finalization_pass(self):
        tool_response = SimpleNamespace(
            content="", thinking="",
            tool_calls=[SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="demo", arguments="{}"),
            )],
        )
        final_response = SimpleNamespace(
            content="正式回答", thinking="", tool_calls=[],
        )
        client = _SequenceClient([tool_response, final_response])

        chunks = list(agent_runner_loop(
            client, "system", "question", _FinalizingHandler(),
            [{"type": "function", "function": {"name": "demo"}}],
            max_turns=1, verbose=True,
        ))

        text = "".join(chunk for chunk in chunks if isinstance(chunk, str))
        self.assertIn("正式回答", text)
        self.assertTrue(client.tool_sets[0])
        self.assertEqual(client.tool_sets[1], [])
        self.assertNotIn("允许的轮次内形成完整答复", text)

    def test_function_tag_tool_call_is_parsed(self):
        calls, content = _parse_text_tool_calls(
            '<summary>准备列文档</summary>\n'
            '<function=kb_list>{"data_id":"doc-1"}</function>'
        )

        self.assertEqual(content, "<summary>准备列文档</summary>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "kb_list")
        self.assertEqual(json.loads(calls[0].function.arguments), {"data_id": "doc-1"})

    def test_empty_function_tag_uses_empty_arguments(self):
        calls, content = _parse_text_tool_calls(
            "<function=kb_list></function>"
        )

        self.assertEqual(content, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "kb_list")
        self.assertEqual(json.loads(calls[0].function.arguments), {})

    def test_malformed_function_arguments_become_recoverable_bad_json(self):
        calls, content = _parse_text_tool_calls(
            "<function=kb_search>{not-json}</function>"
        )

        self.assertEqual(content, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "bad_json")
        self.assertIn(
            "kb_search", json.loads(calls[0].function.arguments)["msg"]
        )

    def test_tool_shaped_text_is_not_a_visible_final_answer(self):
        visible = GenericAgentHandler._visible_response_text(
            "<summary>准备调用</summary><function=kb_list>bad</function>"
        )

        self.assertEqual(visible, "")

    def test_interactive_turn_31_does_not_request_file_checkpoint(self):
        parent = SimpleNamespace(
            knowledge_scope={"mode": "none"}, task_dir=None,
            extrakeyinfo=None, intervene=None, _turn_end_hooks={},
        )
        handler = GenericAgentHandler(parent)
        prompt = handler.turn_end_callback(
            SimpleNamespace(content="<summary>继续处理</summary>"),
            [{"tool_name": "demo", "args": {}}], [], 31, "continue", {},
        )

        self.assertNotIn("file", prompt.lower())
        self.assertNotIn("检查点", prompt)


if __name__ == "__main__":
    unittest.main()
