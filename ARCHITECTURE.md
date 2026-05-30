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
│   ├── warden.py                   # pre-execution authorization policies (deny/confirm/allow)
│   ├── scheduler.py                # TaskStore + Scheduler + schedule()/list/cancel tools
│   ├── store.py                    # SqliteStore (durable session)
│   ├── run_command.py              # allowlisted external commands (+ forbidden-arg / secret-path denylists)
│   ├── evals.py                    # behavioral eval harness (scenarios, assertions, runner, setup)
│   ├── mcp.py                      # McpServer + mcp_tools() — wrap any stdio MCP server
│   ├── env.py                      # load_env, require_env
│   ├── integrations/
│   │   └── hippocamp.py            # MCP server wrapper for Hippocamp memory
│   ├── transports/
│   │   └── telegram.py             # TelegramTransport (polling + streaming edits)
│   └── cli/
│       ├── main.py                 # `ambi` subcommands: init / run / chat / eval / version
│       ├── build.py                # opinionated build_agent() factory
│       ├── paths.py                # ~/.ambi/ layout resolution (AMBI_HOME override)
│       ├── system.md               # bundled default personality
│       └── system_hippocamp.md     # appended when AMBI_USE_HIPPOCAMP=1
├── evals/                          # behavioral scenarios (scenarios/*.yaml), run by `ambi eval`
├── examples/                       # raw library API demos (repl.py, telegram_bot.py)
├── tests/                          # pytest, asyncio-mode auto (305 unit + 3 live smoke)
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
       │      warden.authorize(call)  → deny / confirm / allow │
       │      gather(tools.invoke(c) for c in tool_use)       │
       │      wrap results in <tool_output trust="data"> view │
       │      sensegate.check(final_text, invocations)        │
       │      block-and-retry on writes / flag-only on reads  │
       └──┬─────┬──────┬──────┬─────────┬───────┬───────┬─────┘
          │     │      │      │         │       │       │
          ▼     ▼      ▼      ▼         ▼       ▼       ▼
     ToolRegistry │  Warden provider SkillRegistry SenseGate SqliteStore
                  │    │      │         │            │          │
                  │    ▼      ▼         ▼            ▼          ▼
                  │  policies GoogleP. catalog +   LLM judge  messages
                  │  +audit    │       load_skill  + audit    + tasks
                  │            ▼
              (handlers)  Gemini API
```

The streaming variant (`Agent.chat_stream`) yields events of types `TextDelta`, `ToolUseEvent`, `ToolResultEvent`, `SenseGateFlagEvent`, `ChatComplete` — same loop semantics, same persistence, same rollback. The Telegram transport edits a placeholder message progressively as text arrives; the REPL uses rich `Live` to update the panel.

Before any tool handler runs, `Agent._invoke_with_timeout` asks the Warden to authorize the call; a `deny` (or unconfirmed `require_confirmation`) becomes an error `ToolResultBlock` the model sees instead of execution. Tool results are wrapped in a sanitized `<tool_output trust="data">` envelope in the LLM-facing context view (see *Security* below).

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

Skills are **advisory** (prose the model reads), not authoritative. Authorization lives in the tool layer (`CommandPolicy`, the Warden, MCP transport, etc.). Skills describe *how and when* to use capabilities; they don't enforce *what is allowed*. See the module docstring in `ambi/skills/__init__.py` for the full guidance with ✅/❌ examples.

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

The verifier reads attacker-controllable tool output, so each result is run through `sanitize_tool_output()` and wrapped in a trust envelope before it reaches the judge, and the judge prompt instructs it to never obey instructions embedded in tool results — a poisoned result can't talk the integrity check into rubber-stamping a false claim.

## Warden — pre-execution authorization

Where SenseGate verifies *what happened* after a turn (soft, LLM-based), the **Warden** (`ambi/warden.py`) authorizes *what may happen* before it does (hard, deterministic). It holds an ordered list of policies; each returns `allow` / `deny` / `require_confirmation`; the first non-`allow` wins; every decision (including allows) is appended to `audit_log`. An empty Warden is a no-op (allow everything).

`Agent._invoke_with_timeout` calls `warden.authorize(ctx)` before every tool handler:

- **`deny`** → an error `ToolResultBlock` is returned; the handler never runs.
- **`require_confirmation`** → the agent's optional async `confirm(ctx, decision)` hook is awaited. **Fail-closed:** if no confirmer is wired, or it returns False or raises, the call is declined (error result, not executed). `ambi chat` wires an interactive y/N confirmer that pauses the `Live` panel; the headless daemon wires none, so outbound actions fail closed there.

Starter policies: `AllowlistPolicy`, `CommandAllowlistPolicy`, `ArgvValidatorPolicy` (case-insensitive substring **and** ordered-token-subsequence matching, so injected flags can't evade), `UrlAllowlistPolicy` (denies egress to non-allowlisted hosts found in argv — the exfil cut), `RequireConfirmationPolicy`, `CostCeilingPolicy`, `QuietHoursPolicy`. The default stack is assembled in `ambi/cli/build.py`.

## Security — prompt-injection defense

The primary threat is prompt injection via untrusted tool output. Defenses are layered across this codebase; the canonical reference is **[SECURITY.md](SECURITY.md)**. In brief:

- **Trust envelope + sanitizer** (`ambi/types.py`): `sanitize_tool_output()` strips invisible/zero-width chars, Unicode "tag" smuggling, bidi overrides, NFKC-folds compatibility forms, and redacts plaintext injection markers; `wrap_tool_output()` wraps results in `<tool_output trust="data">`. Applied non-destructively in `Agent`'s context view (storage stays raw), in the SenseGate verifier, and in the compaction summarizer (whose output persists into future turns).
- **System framing** (`ambi/cli/system.md`): tool outputs are data to reason about, never commands — including decoded/encoded payloads.
- **Structural caps** (the Warden + `CommandPolicy`): `run_command` enforces an argv[0] allowlist (no `find`), forbidden-arg blocking (`-exec`/`-delete`), and a secret-path denylist (`.env`, `~/.ssh`, …); the Warden adds egress allowlisting and fail-closed confirmation. These hold even when the prompt-level defenses are bypassed.

## CLI

`ambi` is the installed entry point. Subcommands:

| Command | Behaviour |
|---|---|
| `ambi init` | Seed `~/.ambi/` with `.env` template + `system.md`. Bundled skills are not copied — they ship with the package. |
| `ambi chat` | Local REPL. Rich panels, prompt_toolkit input (bracketed paste, history), streaming. |
| `ambi run` | Telegram daemon + scheduler. Persistent session shared with `ambi chat`. |
| `ambi eval [path]` | Run behavioral scenarios (default `evals/scenarios/`) against a real provider; per-scenario panel + summary table; exit non-zero on any failure. See *Evals*. |
| `ambi version` | Print version. |

Env layering: `~/.ambi/.env` (defaults) → project-local `./.env` (overrides). Project file wins on conflicting keys; home file fills the rest.

## Evals — behavioral testing

Unit tests verify the harness; **evals** verify *behavior* — prompt-level regressions pytest can't see (e.g. "after editing `system.md`, does the agent still answer general knowledge without firing `recall_memory`?"). Lives in `ambi/evals.py`, driven by `ambi eval`.

A **scenario** (`evals/scenarios/*.yaml`) is a `Scenario(input, assertions, setup)`:

```yaml
name: tool_followup_text
description: After a tool call, the agent must still produce text.
input: "What time is it in Tokyo right now?"
assert:
  - tool_called: get_current_time
  - text_matches: '(?i)tokyo|jst|asia'
  - text_not_matches: '^\s*$'
```

**Runner** (`run_scenario`): drives `agent.chat_stream(input)`, collecting `ToolUseEvent` names and the final `ChatComplete` text, then evaluates each assertion into an `AssertionResult`; a `ScenarioResult` passes iff no error and all assertions pass. Token/cost usage is attributed to the run by diffing `_usage_snapshot(agent)` before and after — it reads the `TrackingProvider`'s in-memory accumulator (`usage_snapshot()`), so the `max_*_tokens` / `max_cost_usd` assertions enforce real numbers and the report shows per-scenario tokens + cost. The CLI runs each scenario with a fresh `build_agent()` (no Hippocamp, no task store), retries empty-stream flakes up to `AMBI_EVAL_MAX_ATTEMPTS` (default 2), and renders a per-scenario panel + summary table.

**Assertion types** (`check_assertion`): `text_contains` / `text_not_contains` (case-insensitive substring), `text_matches` / `text_not_matches` (regex), `tool_called` / `tool_not_called`, and `max_input_tokens` / `max_output_tokens` / `max_cost_usd`.

**Setup blocks** (`apply_scenario_setup`, a context manager): seed per-scenario state in a temp dir, restored on exit. `setup.env` overrides environment variables (with `{tmp_dir}` substitution); `setup.prepare` runs registered actions. The built-in `create_obsidian_notes` seeds a vault; add your own via `register_prepare_action(name, fn)`.

```yaml
setup:
  env: { OBSIDIAN_VAULT: "{tmp_dir}/vault" }
  prepare:
    - create_obsidian_notes: { count: 150, folders: [Inbox, Areas/Work] }
```

## Concurrency

Tool calls within a single turn run in parallel via `asyncio.gather`. The order of `ToolResultBlock`s in the resulting user message preserves the order the model emitted them in (zip-by-position), so `tool_use_id`s line up correctly.

`Agent._chat_lock` serializes any concurrent `chat()` callers (Telegram + scheduler firing simultaneously) — history never interleaves between triggers.
