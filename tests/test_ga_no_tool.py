import types
import unittest

import ga


def _run(gen):
    out = []
    try:
        while True:
            out.append(next(gen))
    except StopIteration as e:
        return out, e.value


class GANoToolTests(unittest.TestCase):
    def _handler(self):
        parent = types.SimpleNamespace(verbose=False, task_dir=None)
        return ga.GenericAgentHandler(parent, cwd="/home/runner/work/GenericAgent/GenericAgent")

    def test_unfinished_no_tool_reply_keeps_loop_running(self):
        handler = self._handler()
        response = types.SimpleNamespace(content="I'll inspect the codebase and then update the fix.", thinking="")
        logs, outcome = _run(handler.do_no_tool({}, response))
        self.assertIn("[Info] Reply looks unfinished. Continue asking.\n", logs)
        self.assertIsNotNone(outcome.next_prompt)
        self.assertIsNone(outcome.data)

    def test_real_direct_answer_can_finish(self):
        handler = self._handler()
        response = types.SimpleNamespace(content="The task is complete and the GUI now waits for completion before showing the final response.", thinking="")
        logs, outcome = _run(handler.do_no_tool({}, response))
        self.assertIn("[Info] Final response to user.\n", logs)
        self.assertIsNone(outcome.next_prompt)
        self.assertIs(outcome.data, response)


if __name__ == "__main__":
    unittest.main()
