import io
import sys
import types
import unittest
from unittest.mock import patch

import llmcore


class CopilotSDKSessionTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {}
        self.record = {}
        self._install_copilot_stubs()

    def tearDown(self):
        for name, mod in self._saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def _swap_module(self, name, module):
        if name not in self._saved_modules:
            self._saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    def _install_copilot_stubs(self):
        record = self.record

        class FakeSubprocessConfig:
            def __init__(self, **kwargs):
                record["subprocess_kwargs"] = kwargs
                self.kwargs = kwargs

        class FakeSession:
            def __init__(self):
                self._handlers = []

            def on(self, handler):
                self._handlers.append(handler)

                def unsubscribe():
                    if handler in self._handlers:
                        self._handlers.remove(handler)

                return unsubscribe

            async def send_and_wait(self, prompt, **kwargs):
                record["send_and_wait_calls"] = record.get("send_and_wait_calls", 0) + 1
                record["send_and_wait_kwargs"] = {"prompt": prompt, **kwargs}
                remaining = int(record.get("fail_send_and_wait_times", 0) or 0)
                if remaining > 0:
                    record["fail_send_and_wait_times"] = remaining - 1
                    raise RuntimeError("transient send failure")
                record["prompt"] = prompt
                for delta in record.get("stream_chunks", []):
                    evt = types.SimpleNamespace(data=types.SimpleNamespace(delta_content=delta))
                    for handler in list(self._handlers):
                        handler(evt)
                if record.get("fail_with_idle_timeout"):
                    raise TimeoutError("Timeout after 30.0s waiting for session.idle")
                reply_content = record.get("reply_content", "stubbed copilot reply")
                evt = types.SimpleNamespace(data=types.SimpleNamespace(content=reply_content))
                for handler in list(self._handlers):
                    handler(evt)
                return types.SimpleNamespace(data=types.SimpleNamespace(content=reply_content))

            async def disconnect(self):
                record["disconnected"] = True

        class FakeCopilotClient:
            def __init__(self, config=None, **kwargs):
                record["client_config"] = config
                record["client_kwargs"] = kwargs
                self._client = types.SimpleNamespace(
                    get_stderr_output=lambda: record.get("stderr_output", "")
                )

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def create_session(self, **kwargs):
                record["create_session_kwargs"] = kwargs
                return FakeSession()

        copilot_mod = types.ModuleType("copilot")
        copilot_mod.CopilotClient = FakeCopilotClient
        copilot_mod.SubprocessConfig = FakeSubprocessConfig

        session_mod = types.ModuleType("copilot.session")

        class PermissionHandler:
            approve_all = object()

        session_mod.PermissionHandler = PermissionHandler
        self._swap_module("copilot", copilot_mod)
        self._swap_module("copilot.session", session_mod)

    def test_resolve_session_supports_copilot_sdk_config(self):
        cfg = {
            "name": "copilot-sdk",
            "model": "gpt-5",
            "github_token": "ghp_test",
            "copilot_home": "/tmp/copilot-home",
            "cli_args": ["--dummy-flag"],
            "cli_log_level": "debug",
        }
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            self.assertIsInstance(session, llmcore.CopilotSDKSession)
            output = "".join(session.ask("hello copilot sdk"))

        self.assertIn("stubbed copilot reply", output)
        self.assertEqual(self.record["create_session_kwargs"]["model"], "gpt-5")
        self.assertTrue(self.record["create_session_kwargs"]["streaming"])
        self.assertIn("on_permission_request", self.record["create_session_kwargs"])
        self.assertEqual(self.record["subprocess_kwargs"]["github_token"], "ghp_test")
        self.assertTrue(self.record.get("disconnected"))

    def test_resolve_client_wraps_copilot_sdk_as_tool_client(self):
        cfg = {"model": "gpt-5"}
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            client = llmcore.resolve_client("copilot_sdk_config")
        self.assertIsInstance(client, llmcore.ToolClient)

    def test_copilot_sdk_logs_cli_output_to_console(self):
        cfg = {"model": "gpt-5"}
        self.record["stderr_output"] = "copilot cli debug log\n"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("stubbed copilot reply", output)
        self.assertIn("copilot cli debug log", stderr.getvalue())

    def test_copilot_sdk_can_disable_cli_console_logs(self):
        cfg = {"model": "gpt-5", "cli_log_to_console": False}
        self.record["stderr_output"] = "copilot cli debug log\n"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("stubbed copilot reply", output)
        self.assertEqual("", stderr.getvalue())

    def test_copilot_sdk_ask_retries_transient_send_failures(self):
        cfg = {"model": "gpt-5", "max_retries": 2, "base_delay": 0}
        self.record["fail_send_and_wait_times"] = 1
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("stubbed copilot reply", output)
        self.assertNotIn("!!!Error:", output)
        self.assertEqual(2, self.record.get("send_and_wait_calls"))

    def test_copilot_sdk_ask_emits_error_after_retry_exhaustion(self):
        cfg = {"model": "gpt-5", "max_retries": 1, "base_delay": 0}
        self.record["fail_send_and_wait_times"] = 5
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("!!!Error: RuntimeError: transient send failure", output)
        self.assertEqual(2, self.record.get("send_and_wait_calls"))

    def test_copilot_sdk_stream_mode_yields_incremental_chunks(self):
        cfg = {"model": "gpt-5"}
        self.record["stream_chunks"] = ["stubbed ", "copilot ", "reply"]
        self.record["reply_content"] = "stubbed copilot reply"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            chunks = list(session.ask("hello copilot sdk"))
        self.assertEqual(["stubbed ", "copilot ", "reply"], chunks)
        self.assertEqual("stubbed copilot reply", "".join(chunks))
        self.assertTrue(self.record["create_session_kwargs"]["streaming"])

    def test_copilot_sdk_non_stream_mode_returns_plain_text(self):
        cfg = {"model": "gpt-5", "stream": False}
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = session.ask("hello copilot sdk")
        self.assertEqual("stubbed copilot reply", output)
        self.assertFalse(self.record["create_session_kwargs"]["streaming"])


    def test_copilot_sdk_idle_timeout_ignored_no_content(self):
        """session.idle TimeoutError with no streamed content must not surface as !!!Error:."""
        cfg = {"model": "gpt-5", "max_retries": 0, "base_delay": 0}
        self.record["fail_with_idle_timeout"] = True
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = "".join(session.ask("hello"))
        self.assertNotIn("!!!Error:", output)
        self.assertEqual("", output)

    def test_copilot_sdk_idle_timeout_ignored_with_partial_content(self):
        """session.idle TimeoutError after partial streamed content must not append !!!Error:."""
        cfg = {"model": "gpt-5", "max_retries": 0, "base_delay": 0}
        self.record["stream_chunks"] = ["partial ", "reply"]
        self.record["fail_with_idle_timeout"] = True
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = "".join(session.ask("hello"))
        self.assertNotIn("!!!Error:", output)
        self.assertEqual("partial reply", output)

    def test_copilot_sdk_idle_timeout_ignored_non_stream_mode(self):
        """session.idle TimeoutError in non-stream mode must not surface as !!!Error:."""
        cfg = {"model": "gpt-5", "stream": False, "max_retries": 0, "base_delay": 0}
        self.record["fail_with_idle_timeout"] = True
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = session.ask("hello")
        self.assertNotIn("!!!Error:", output)
        self.assertEqual("", output)


if __name__ == "__main__":
    unittest.main()
