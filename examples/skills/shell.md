---
name: shell
description: When the user asks about files, directories, or git state in the current project.
---

To answer questions about the local filesystem or git, call `run_command`
with `argv` as a list — never a single shell string. The tool description
lists which commands are currently allowed; if a command isn't allowed,
say so honestly rather than guessing.

Examples:
- list files:     `run_command({"argv": ["ls", "-la"]})`
- inspect repo:   `run_command({"argv": ["git", "status"]})`
- recent commits: `run_command({"argv": ["git", "log", "--oneline", "-n", "10"]})`
- show a file:    `run_command({"argv": ["cat", "README.md"]})`
- search:         `run_command({"argv": ["grep", "-r", "pattern", "."]})`

Use `cwd` to target a subdirectory when relevant.
