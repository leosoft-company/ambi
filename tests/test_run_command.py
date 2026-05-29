import sys
from pathlib import Path

import pytest

from ambi.run_command import CommandPolicy, make_run_command_tool

PY = sys.executable


async def test_argv_must_be_list_of_strings():
    tool = make_run_command_tool(CommandPolicy(allowed={"echo"}))
    assert "argv must be" in await tool.handler({})
    assert "argv must be" in await tool.handler({"argv": "echo hi"})
    assert "argv must be" in await tool.handler({"argv": []})
    assert "argv must be" in await tool.handler({"argv": ["echo", 1]})


async def test_command_not_in_allowlist():
    tool = make_run_command_tool(CommandPolicy(allowed={"git"}))
    result = await tool.handler({"argv": ["rm", "-rf", "/"]})
    assert "'rm'" in result
    assert "not allowlisted" in result
    assert "git" in result  # surfaces the allowed list


async def test_allowlist_matches_basename():
    tool = make_run_command_tool(CommandPolicy(allowed={"python3"}))
    # Even with absolute path, basename "python3" should be checked.
    result = await tool.handler({"argv": ["/usr/bin/python3-not-real", "-V"]})
    # The basename "python3-not-real" is not allowlisted -> error.
    assert "not allowlisted" in result


async def test_successful_command_returns_exit_code_and_stdout():
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=10),
    )
    result = await tool.handler(
        {"argv": [PY, "-c", "print('hello world')"]}
    )
    assert "exit_code: 0" in result
    assert "hello world" in result


async def test_nonzero_exit_code_surfaced():
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=10),
    )
    result = await tool.handler(
        {"argv": [PY, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(7)"]}
    )
    assert "exit_code: 7" in result
    assert "boom" in result  # stderr captured


async def test_timeout_kills_process():
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=0.1),
    )
    result = await tool.handler(
        {"argv": [PY, "-c", "import time; time.sleep(5)"]}
    )
    assert "timed out" in result


async def test_command_not_found():
    tool = make_run_command_tool(
        CommandPolicy(allowed={"definitely_not_a_real_binary_xyz"}),
    )
    result = await tool.handler({"argv": ["definitely_not_a_real_binary_xyz"]})
    assert "not found" in result


async def test_output_truncation():
    tool = make_run_command_tool(
        CommandPolicy(
            allowed={Path(PY).name},
            default_timeout=10,
            max_output_bytes=50,
        ),
    )
    result = await tool.handler(
        {"argv": [PY, "-c", "print('x' * 200)"]}
    )
    assert "truncated" in result


async def test_cwd_must_be_under_root(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path.parent  # one level above tmp_path's root

    tool = make_run_command_tool(
        CommandPolicy(
            allowed={Path(PY).name},
            cwd_root=tmp_path,
            default_timeout=10,
        ),
    )
    ok = await tool.handler(
        {"argv": [PY, "-c", "print('ok')"], "cwd": str(inside)}
    )
    assert "exit_code: 0" in ok

    bad = await tool.handler(
        {"argv": [PY, "-c", "print('ok')"], "cwd": str(outside)}
    )
    assert "must be under" in bad


async def test_cwd_missing(tmp_path):
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=10),
    )
    bogus = str(tmp_path / "does-not-exist")
    result = await tool.handler({"argv": [PY, "-c", "print('x')"], "cwd": bogus})
    assert "not a directory" in result




# ---------- streaming progress ----------


async def test_progress_callback_receives_stdout_lines():
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=10),
    )
    progress_msgs: list[str] = []

    async def progress(msg):
        progress_msgs.append(msg)

    await tool.handler(
        {
            "argv": [
                PY, "-c",
                "print('one'); print('two'); print('three')",
            ],
        },
        progress=progress,
    )
    assert "one" in progress_msgs
    assert "two" in progress_msgs
    assert "three" in progress_msgs


async def test_progress_cap_caps_emissions_and_reports_dropped():
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=10),
    )
    progress_msgs: list[str] = []

    async def progress(msg):
        progress_msgs.append(msg)

    # 150 lines — cap is 100; expect dropped notification
    await tool.handler(
        {
            "argv": [
                PY, "-c",
                "for i in range(150): print(f'line{i}')",
            ],
        },
        progress=progress,
    )
    line_msgs = [m for m in progress_msgs if m.startswith("line")]
    assert len(line_msgs) == 100  # exactly capped
    dropped = [m for m in progress_msgs if "more lines collected" in m]
    assert len(dropped) == 1
    assert "50" in dropped[0]  # 150 - 100 = 50 dropped


async def test_stderr_lines_also_stream_with_prefix():
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=10),
    )
    progress_msgs: list[str] = []

    async def progress(msg):
        progress_msgs.append(msg)

    await tool.handler(
        {
            "argv": [
                PY, "-c",
                "import sys; sys.stderr.write('boom\\n')",
            ],
        },
        progress=progress,
    )
    err_lines = [m for m in progress_msgs if m.startswith("stderr:")]
    assert err_lines
    assert "boom" in err_lines[0]


async def test_handler_works_without_progress_callback():
    """The chat() path passes no progress — must still work."""
    tool = make_run_command_tool(
        CommandPolicy(allowed={Path(PY).name}, default_timeout=10),
    )
    result = await tool.handler({"argv": [PY, "-c", "print('hi')"]})
    assert "exit_code: 0" in result
    assert "hi" in result
