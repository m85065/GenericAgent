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
            async def send_and_wait(self, prompt):
                record["prompt"] = prompt
                on_event = record.get("_on_event")
                message = record.get("message_content", "stubbed copilot reply")
                final_message = record.get("final_message_content", message)
                if on_event:
                    # Fire assistant.message_delta (streaming content)
                    on_event(types.SimpleNamespace(
                        type=types.SimpleNamespace(value="assistant.message_delta"),
                        data=types.SimpleNamespace(delta_content=message),
                    ))
                    # Fire optional tool execution progress event
                    event_progress = record.get("event_progress_output")
                    if event_progress:
                        on_event(types.SimpleNamespace(
                            type=types.SimpleNamespace(value="tool.execution_progress"),
                            data=types.SimpleNamespace(progress_message=event_progress),
                        ))
                if record.get("raise_timeout"):
                    raise TimeoutError("Timeout after 60.0s waiting for session.idle")
                return types.SimpleNamespace(data=types.SimpleNamespace(content=final_message))

            async def disconnect(self):
                record["disconnected"] = True

        class FakeCopilotClient:
            def __init__(self, config=None, **kwargs):
                record["client_config"] = config
                record["client_kwargs"] = kwargs
                self._client = types.SimpleNamespace(
                    get_stderr_output=lambda: record.get("stderr_output", ""),
                    get_progress_output=lambda: record.get("progress_output", ""),
                )
            def get_stderr_output(self):
                return record.get("outer_stderr_output", "")
            def get_progress_output(self):
                return record.get("outer_progress_output", "")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def create_session(self, **kwargs):
                record["create_session_kwargs"] = kwargs
                record["_on_event"] = kwargs.get("on_event")
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
            with patch("sys.stderr", new_callable=io.StringIO):
                output = "".join(session.ask("hello copilot sdk"))

        self.assertIn("stubbed copilot reply", output)
        self.assertEqual(self.record["create_session_kwargs"]["model"], "gpt-5")
        self.assertIn("on_permission_request", self.record["create_session_kwargs"])
        self.assertTrue(self.record["create_session_kwargs"].get("streaming"))
        self.assertIsNotNone(self.record["create_session_kwargs"].get("on_event"))
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

    def test_copilot_sdk_logs_cli_progress_output_to_console(self):
        cfg = {"model": "gpt-5"}
        self.record["progress_output"] = "copilot cli progress: planning...\n"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("stubbed copilot reply", output)
        self.assertIn("copilot cli progress: planning...", stderr.getvalue())

    def test_copilot_sdk_logs_outer_client_progress_output_to_console(self):
        cfg = {"model": "gpt-5"}
        self.record["outer_progress_output"] = "copilot outer progress log\n"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("stubbed copilot reply", output)
        self.assertIn("copilot outer progress log", stderr.getvalue())

    def test_copilot_sdk_streaming_delta_forwarded_to_stderr(self):
        cfg = {"model": "gpt-5"}
        self.record["message_content"] = "Hello from Copilot!"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("Hello from Copilot!", output)
        self.assertIn("Hello from Copilot!", stderr.getvalue())

    def test_copilot_sdk_streaming_multi_delta_accumulates_correctly(self):
        """Multiple streaming delta events should be joined in order for both output and stderr."""
        cfg = {"model": "gpt-5"}
        record = self.record

        class MultiDeltaSession:
            async def send_and_wait(self, prompt):
                on_event = record.get("_on_event")
                if on_event:
                    for chunk in ["Hello", " from", " Copilot!"]:
                        on_event(types.SimpleNamespace(
                            type=types.SimpleNamespace(value="assistant.message_delta"),
                            data=types.SimpleNamespace(delta_content=chunk),
                        ))
                return types.SimpleNamespace(data=types.SimpleNamespace(content=""))

            async def disconnect(self):
                record["disconnected"] = True

        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            copilot_mod = sys.modules["copilot"]
            orig_cls = copilot_mod.CopilotClient

            class PatchedClient(orig_cls):
                async def create_session(self, **kwargs):
                    record["_on_event"] = kwargs.get("on_event")
                    return MultiDeltaSession()

            with patch.dict(sys.modules, {"copilot": type(copilot_mod)("copilot")}):
                sys.modules["copilot"].CopilotClient = PatchedClient
                sys.modules["copilot"].SubprocessConfig = copilot_mod.SubprocessConfig
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    output = "".join(session.ask("hello copilot sdk"))

        self.assertEqual("Hello from Copilot!", output)
        self.assertIn("Hello", stderr.getvalue())
        self.assertIn(" from", stderr.getvalue())
        self.assertIn(" Copilot!", stderr.getvalue())

    def test_copilot_sdk_prefers_stream_deltas_over_final_message_fallback(self):
        cfg = {"model": "gpt-5"}
        self.record["message_content"] = "delta-only"
        self.record["final_message_content"] = "final-message"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("delta-only", output)
        self.assertNotIn("final-message", output)

    def test_copilot_sdk_logs_progress_from_tool_execution_events(self):
        cfg = {"model": "gpt-5"}
        self.record["event_progress_output"] = "copilot sdk progress event\n"
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("stubbed copilot reply", output)
        self.assertIn("copilot sdk progress event", stderr.getvalue())

    def test_copilot_sdk_session_idle_timeout_uses_streamed_partial(self):
        cfg = {"model": "gpt-5"}
        self.record["message_content"] = "partial before timeout"
        self.record["raise_timeout"] = True
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("partial before timeout", output)
        self.assertNotIn("!!!Error:", output)

    def test_copilot_sdk_session_idle_timeout_without_stream_shows_warning(self):
        cfg = {"model": "gpt-5"}
        self.record["message_content"] = ""
        self.record["raise_timeout"] = True
        with patch.object(llmcore, "reload_mykeys", return_value=({"copilot_sdk_config": cfg}, True)):
            session = llmcore.resolve_session("copilot_sdk_config")
            output = "".join(session.ask("hello copilot sdk"))
        self.assertIn("[WARN] Copilot session idle timeout", output)
        self.assertNotIn("!!!Error:", output)


if __name__ == "__main__":
    unittest.main()
