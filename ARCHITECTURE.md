# ambi-core architecture

## Package layout

```
ambi-core/
├── ambi/
│   ├── types.py                    # Message, Block (Text/ToolUse/ToolResult),
│   │                               #   CompletionResult, ToolDef + streaming events
│   ├── tool.py                     # Tool, ToolRegistry, ToolKind (read/write)
│   ├── skills/                     # ← bundled skills (self-contained packages)
│   │   ├── __init__.py             #   SkillRegistry, load_skill tool, register_bundled_skill_tools
│   │   ├── time/SKILL.md
│   │   ├── shell/SKILL.md
│   │   └── obsidian/
│   │       ├── SKILL.md            #   prose guidance for the model
│   │       └── tools.py            #   implementation + register(tools) entry point
│   ├── provider.py                 # LLMProvider Protocol — complete + stream
│   ├── providers/google.py         # Gemini adapter (google-genai)
│   ├── loop.py                     # Agent (chat + chat_stream), MaxTurnsExceeded
│   ├── sensegate.py                # post-turn claim verifier (LLM judge + audit log)
│   ├── scheduler.py                # TaskStore + Scheduler + schedule()/list/cancel tools
│   ├── store.py                    # SqliteStore (durable session)
│   ├── run_command.py              # allowlisted external commands
│   ├── mcp.py                      # McpServer + mcp_tools() — wrap any stdio MCP server
│   ├── env.py                      # load_env, require_env
│   ├── integrations/
│   │   └── hippocamp.py            # MCP server wrapper for Hippocamp memory
│   ├── transports/
│   │   └── telegram.py             # TelegramTransport (polling + streaming edits)
│   └── cli/
│       ├── main.py                 # `ambi` subcommands: init / run / chat / version
│       ├── build.py                # opinionated build_agent() factory
│       ├── paths.py                # ~/.ambi/ layout resolution (AMBI_HOME override)
│       ├── system.md               # bundled default personality
│       └── system_hippocamp.md     # appended when AMBI_USE_HIPPOCAMP=1
├── examples/                       # raw library API demos (repl.py, telegram_bot.py)
├── tests/                          # pytest, asyncio-mode auto (180 unit + 3 live smoke)
├── docs/                           # demo SVG + render script
└── pyproject.toml                  # hatchling build, ambi script + [hippocamp] extra
```

## Component graph

```
                ┌──────────────────────────────┐
                │  CLI / Transport             │
                │  ambi chat · ambi run · ...  │
                └──────────────┬───────────────┘
                               │ agent.chat(text) / chat_stream(text)
                               ▼
       ┌──────────────────────────────────────────────────────┐
       │                        Agent                         │
       │  - messages       (durable session, via SqliteStore) │
       │  - system         (personality + skill catalog)      │
       │  - chat() loop:                                      │
       │      provider.complete/stream → assistant blocks     │
       │      gather(tools.invoke(c) for c in tool_use)       │
       │      sensegate.check(final_text, invocations)        │
       │      block-and-retry on writes / flag-only on reads  │
       └──┬───────┬──────┬───────┬──────────┬───────┬─────────┘
          │       │      │       │          │       │
          ▼       ▼      ▼       ▼          ▼       ▼
     ToolRegistry │   provider  SkillRegistry  SenseGate  SqliteStore
                  │      │         │              │           │
                  │      ▼         ▼              ▼           ▼
                  │  GoogleProvider  catalog +   LLM judge   messages
                  │      │           load_skill  + audit_log + tasks
                  │      ▼
              (handlers) Gemini API
```

The streaming variant (`Agent.chat_stream`) yields events of types `TextDelta`, `ToolUseEvent`, `ToolResultEvent`, `SenseGateFlagEvent`, `ChatComplete` — same loop semantics, same persistence, same rollback. The Telegram transport edits a placeholder message progressively as text arrives; the REPL uses rich `Live` to update the panel.

## Provider seam

The `LLMProvider` Protocol is the only contract:

```
async def complete(messages, tools, system, max_tokens, **kw) -> CompletionResult
def stream(messages, tools, system, max_tokens, **kw) -> AsyncIterator[ProviderChunk]
```

`messages` and `tools` are in the normalized shape (Anthropic-influenced: `ToolResultBlock` lives inside `user` messages). Each adapter translates in/out:

```
              normalized                       provider-native
  ┌─────────────────────────────┐    ┌──────────────────────────────┐
  │ Message(role, [Blocks])     │ ⇄  │ gt.Content(role, [Parts])    │
  │ ToolUseBlock(id, name, in)  │ ⇄  │ gt.Part(function_call=...)   │
  │ ToolResultBlock(id, content)│ ⇄  │ gt.Part(function_response=…) │
  │ ToolDef(name, desc, schema) │ ⇄  │ gt.FunctionDeclaration(...)  │
  │ CompletionResult(stop, ...) │ ⇄  │ resp.candidates[0]           │
  └─────────────────────────────┘    └──────────────────────────────┘
```

Gemini-specific quirks the adapter absorbs:

- No `tool_use_id` — matched by name; we stash `_tool_name` on `ToolResultBlock` and use it on translation.
- `function_response.response` must be a dict — wrap `{"result": ...}` / `{"error": ...}`.
- System prompt goes in `config`, not the message stream.
- JSON Schema sanitized (`$schema`, `additionalProperties`, etc. stripped) before passing as `gt.Schema`.
- `FinishReason` compared by enum *name* string for SDK forward-compat.

## Skills — progressive disclosure as self-contained packages

A **skill** is a directory under `ambi/skills/<name>/` containing at minimum a `SKILL.md` (YAML frontmatter + prose guidance). Optionally a `tools.py` colocated alongside provides the implementation, with a `register(tools_registry)` entry point that the agent calls at startup.

```
ambi/skills/
  time/
    SKILL.md                       ← prose only; uses cli's built-in get_current_time
    __init__.py
  shell/
    SKILL.md                       ← prose only; uses cli's built-in run_command
    __init__.py
  obsidian/
    SKILL.md                       ← prose guidance (PARA, default-to-Inbox)
    tools.py                       ← obsidian_save / list / search / read / delete
    __init__.py                    ←   + def register(tools): wires them if OBSIDIAN_VAULT set
```

Loading order at `Agent.__init__`:

1. `SkillRegistry.from_dirs(bundled, user)` walks bundled (`ambi/skills/<name>/SKILL.md`) and user (`~/.ambi/skills/`) — later dirs shadow earlier ones on name conflict.
2. `assemble_system()` splices `SKILL CATALOG:\n- name: description\n...` into the system prompt.
3. `register_bundled_skill_tools(tools_registry)` iterates `ambi/skills/*/tools.py` and calls each `register()` entry point. Each skill self-decides whether it's configured (e.g. `obsidian/tools.py:register` no-ops if `OBSIDIAN_VAULT` isn't set).
4. The built-in `load_skill(name)` tool is registered, bound to the registry.

Runtime flow: model sees the catalog in the system prompt → picks a skill → calls `load_skill("obsidian")` → tool returns the SKILL.md body → body enters history as a `ToolResultBlock` → model acts on it using the tools the skill wired up.

### Authoring conventions

Skills are **advisory** (prose the model reads), not authoritative. Authorization lives in the tool layer (`CommandPolicy`, MCP transport, etc.). Skills describe *how and when* to use capabilities; they don't enforce *what is allowed*. See the module docstring in `ambi/skills/__init__.py` for the full guidance with ✅/❌ examples.

## Extending — adding a new skill

Two-minute walkthrough for a hypothetical `weather` skill:

1. **Create the directory**:

   ```
   ambi/skills/weather/
     __init__.py        (empty)
     SKILL.md
     tools.py
   ```

2. **`SKILL.md`** — prose for the model:

   ```markdown
   ---
   name: weather
   description: When the user asks about current weather or forecasts.
   ---

   Use `get_weather(location)` to fetch live conditions. Always return
   temperature, condition, and a one-line outlook — no padding.
   ```

3. **`tools.py`** — implementation + `register()`:

   ```python
   import os, sys
   from ambi.tool import Tool, ToolRegistry
   from ambi.types import ToolDef

   async def _get_weather(args: dict) -> str:
       loc = args.get("location", "").strip()
       # ... hit a weather API ...
       return f"{loc}: 18°C, partly cloudy."

   def register(tools: ToolRegistry) -> None:
       if not os.getenv("WEATHER_API_KEY"):
           return
       tools.register(Tool(
           definition=ToolDef(
               name="get_weather",
               description="Get current weather for a location.",
               input_schema={
                   "type": "object",
                   "properties": {"location": {"type": "string"}},
                   "required": ["location"],
               },
           ),
           handler=_get_weather,
           kind="read",
       ))
   ```

That's the whole change. No edits to `build.py`, the registry, or the agent — `register_bundled_skill_tools` discovers and wires it automatically. Users with `WEATHER_API_KEY` set get the tool; users without get a dormant skill that the model knows exists but won't call.

### User-side overrides

`~/.ambi/skills/<name>/SKILL.md` (or a flat `<name>.md` for prose-only) shadows the bundled package skill of the same name. Users can't add Python tools without modifying the installed package — that's intentional security boundary.

## Persistence

`SqliteStore` (`~/.ambi/data/session.db`) holds the full conversation history. `Agent.load()` populates `messages` at startup; `chat()` appends after every successful turn. A chat() that raises rolls back in memory and never touches disk.

`TaskStore` (`~/.ambi/data/tasks.db`) holds scheduled tasks. The `Scheduler` polls it every 15 seconds and fires due tasks through `agent.chat()`; recurring tasks (`cron`) advance their `run_at` after each fire. The Telegram transport's `on_result` callback delivers the reply as a DM.

## Context window

`Agent.chat()` sends a *trimmed* slice of `self.messages` to the provider on every turn — the last `context_window_turns` user-text turns plus per-block content clipping at `max_block_chars`. The full history stays in memory and on disk; only the LLM-facing slice is compacted. Cutoff logic walks backward looking for user messages whose first block is a `TextBlock` (real user inputs, not tool-result wrappers), so the slice never orphans a `tool_use` from its matching `tool_result`.

## SenseGate

After every turn that involved tool calls, `Agent.chat()` passes `(final_text, invocations)` to `SenseGate.check()`. The default `LLMClaimVerifier` runs a cheap Flash call (thinking disabled, `max_tokens=256`) judging whether the prose accurately describes the tool results. Verdict:

- **Match** → return the text, persist.
- **Mismatch + writes present** → inject a `correction_message` into history, retry (capped at `max_retries=2`). The user sees the corrected version.
- **Mismatch + reads only** → log to `audit_log`, return the text as-is (configurable via `verify_reads=False` to skip the LLM call entirely on read-only turns — cheaper).

## CLI

`ambi` is the installed entry point. Subcommands:

| Command | Behaviour |
|---|---|
| `ambi init` | Seed `~/.ambi/` with `.env` template + `system.md`. Bundled skills are not copied — they ship with the package. |
| `ambi chat` | Local REPL. Rich panels, prompt_toolkit input (bracketed paste, history), streaming. |
| `ambi run` | Telegram daemon + scheduler. Persistent session shared with `ambi chat`. |
| `ambi version` | Print version. |

Env layering: `~/.ambi/.env` (defaults) → project-local `./.env` (overrides). Project file wins on conflicting keys; home file fills the rest.

## Concurrency

Tool calls within a single turn run in parallel via `asyncio.gather`. The order of `ToolResultBlock`s in the resulting user message preserves the order the model emitted them in (zip-by-position), so `tool_use_id`s line up correctly.

`Agent._chat_lock` serializes any concurrent `chat()` callers (Telegram + scheduler firing simultaneously) — history never interleaves between triggers.
