from types import SimpleNamespace

import json

from agent.copilot_acp_client import _extract_tool_calls_from_text, _format_messages_as_prompt


def test_prompt_includes_assistant_tool_calls_without_content():
    prompt = _format_messages_as_prompt(
        [
            {"role": "user", "content": "Investigate this task."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "delegate_task",
                            "arguments": "{\"goal\":\"Inspect the flow\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "tool_name": "delegate_task",
                "content": "{\"results\":[{\"status\":\"ok\"}]}",
            },
        ]
    )

    assert "Assistant:\n<tool_call>" in prompt
    assert '"name": "delegate_task"' in prompt
    assert '"arguments": "{\\"goal\\":\\"Inspect the flow\\"}"' in prompt
    assert "Tool:\n[tool_call_id=call_1, tool_name=delegate_task]" in prompt


def test_prompt_renders_simplenamespace_tool_calls():
    tool_call = SimpleNamespace(
        id="call_ns",
        function=SimpleNamespace(
            name="search_files",
            arguments="{\"pattern\":\"*.md\"}",
        ),
    )

    prompt = _format_messages_as_prompt(
        [
            {
                "role": "assistant",
                "content": "I'll inspect the notes first.",
                "tool_calls": [tool_call],
            }
        ]
    )

    assert "I'll inspect the notes first." in prompt
    assert "<tool_call>" in prompt
    assert '"id": "call_ns"' in prompt
    assert '"name": "search_files"' in prompt


def test_extract_tool_calls_recovers_multiline_terminal_arguments_as_valid_json():
    raw = """<tool_call>{
      "id":"call_1",
      "type":"function",
      "function":{
        "name":"terminal",
        "arguments":"{\\"command\\":\\"python3 - <<'PY'\\\\nprint('hello')\\\\nPY\\",\\"timeout\\":60}"
      }
    }</tool_call>"""

    tool_calls, cleaned = _extract_tool_calls_from_text(raw)

    assert cleaned == ""
    assert len(tool_calls) == 1
    parsed = json.loads(tool_calls[0].function.arguments)
    assert parsed["timeout"] == 60
    assert parsed["command"] == "python3 - <<'PY'\nprint('hello')\nPY"
