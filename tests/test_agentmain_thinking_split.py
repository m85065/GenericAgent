import unittest

import agentmain


class AgentMainThinkingSplitTests(unittest.TestCase):
    def test_split_without_thinking_keeps_gui_text(self):
        state = {"in_thinking": False, "tail": ""}
        think, gui = agentmain.split_thinking_stream("hello world", state, end=True)
        self.assertEqual("", think)
        self.assertEqual("hello world", gui)

    def test_split_routes_thinking_with_tags_to_console(self):
        state = {"in_thinking": False, "tail": ""}
        think, gui = agentmain.split_thinking_stream("a<thinking>internal</thinking>b", state, end=True)
        self.assertEqual("<thinking>internal</thinking>", think)
        self.assertEqual("ab", gui)

    def test_split_handles_cross_chunk_tags(self):
        state = {"in_thinking": False, "tail": ""}
        t1, g1 = agentmain.split_thinking_stream("a<thin", state)
        t2, g2 = agentmain.split_thinking_stream("king>x</thinking>b", state, end=True)
        self.assertEqual("", t1)
        self.assertEqual("a", g1)
        self.assertEqual("<thinking>x</thinking>", t2)
        self.assertEqual("b", g2)

    def test_split_keeps_unclosed_thinking_in_console(self):
        state = {"in_thinking": False, "tail": ""}
        think, gui = agentmain.split_thinking_stream("a<thinking>x", state, end=True)
        self.assertEqual("<thinking>x", think)
        self.assertEqual("a", gui)

    def test_source_routing(self):
        self.assertFalse(agentmain.source_needs_thinking_split("console"))
        self.assertFalse(agentmain.source_needs_thinking_split("task"))
        self.assertFalse(agentmain.source_needs_thinking_split("reflect"))
        self.assertTrue(agentmain.source_needs_thinking_split("user"))
        self.assertTrue(agentmain.source_needs_thinking_split("telegram"))


if __name__ == "__main__":
    unittest.main()
