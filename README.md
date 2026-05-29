# ambi-core

A provider-agnostic personal-agent harness. It runs a single continuous
conversation against any LLM, with first-class skills, action verification,
external tool execution, MCP integration, persistent history, scheduled
tasks, and Telegram delivery.

Built as the runtime half of a personal-AI stack. Pair it with
[Hippocamp](https://github.com/leosoft-company/hippocamp) for long-term
memory that follows you across hosts.

> Status: alpha. Working end-to-end against Gemini 2.5 Flash; API may
> change before 1.0.

---

## What's in the box

| Layer            | What it does                                                      |
|------------------|-------------------------------------------------------------------|
| Loop             | Stateful `Agent.chat()`. Parallel tool dispatch. Tool timeouts. Max-turn guard. Snapshot/rollback on failure. |
| Skills           | YAML-frontmatter markdown skills. Catalog inlined in the system prompt; bodies loaded on demand via `load_skill`. |
| Tools            | `Tool(definition, handler, kind)` — `kind` distinguishes read vs write for SenseGate. Tools run concurrently. |
| Context window   | Sliding window over user-text turns + per-block clipping. Storage stays full-fidelity; only the LLM slice is trimmed. |
| Persistence      | SQLite-backed session via `SqliteStore`. Load on startup, append on each successful `chat()`. |
| SenseGate        | Post-turn LLM judge that compares the assistant's prose against actual tool results. Block-and-retry on writes; flag-only on reads. Configurable. |
| Scheduling       | Cron + one-shot `schedule()` tool the agent can call itself. Fires via `agent.chat()` so scheduled runs use the same tools and skills. |
| MCP integration  | `McpServer` + `mcp_tools()` wrap any stdio MCP server as ambi `Tool`s. Hippocamp ships as a worked example. |
| Run-command      | Allowlisted external commands via `make_run_command_tool(CommandPolicy)`. Argv-only (no shell), cwd jail, timeout, output cap. |
| Transports       | `TelegramTransport` — polling, allowlist auth, typing indicator, message splitting, reply-context extraction, `/scheduled` command. |

## Quick start

```bash
pip install "ambi-core[hippocamp]"   # drop [hippocamp] if you don't want long-term memory
ambi init                            # creates ~/.ambi/ with .env template + example skills
# edit ~/.ambi/.env — add GEMINI_API_KEY, optionally TELEGRAM_BOT_TOKEN + your user ID

ambi chat                            # local REPL
ambi run                             # daemon: Telegram bot + scheduler, always-on
```

Once `ambi run` is up, the Telegram bot becomes your remote — DM it from your phone, ask for reminders, query memory, run commands. The daemon keeps the SQLite session and scheduled tasks alive across restarts.

### From source

```bash
git clone https://github.com/leosoft-company/ambi.git
cd ambi
uv sync --extra dev --extra hippocamp
cp .env.example .env
uv run python examples/repl.py        # or examples/telegram_bot.py
```

The `examples/` scripts show the raw library API; the installed `ambi` CLI is the opinionated path.

### Minimal in-code use

```python
from ambi import Agent, ToolRegistry, SqliteStore, load_env
from ambi.providers.google import GoogleProvider

load_env()  # populate os.environ from .env

agent = Agent(
    provider=GoogleProvider(model="gemini-2.5-flash"),
    tools=ToolRegistry(),
    system="You are a concise assistant.",
    store=SqliteStore("data/session.db"),
)
await agent.load()  # picks up prior session if any
reply = await agent.chat("hello")
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for component graph, runtime
sequence, and the provider seam contract. Short version:

```
caller ─► Agent ─► LLMProvider (Protocol) ─► GoogleProvider ─► Gemini
            │
            ├─► ToolRegistry  (user tools + auto-registered load_skill / scheduler / mcp)
            ├─► SkillRegistry (catalog spliced into system prompt)
            ├─► SenseGate     (post-turn claim verifier)
            └─► SqliteStore   (durable session)
```

## Adding more LLM providers

Implement the `LLMProvider` protocol (`complete(messages, tools, system, **kw) -> CompletionResult`). The only adapter shipped is `GoogleProvider` for Gemini via `google-genai`. Anthropic/OpenAI/local-model adapters slot in the same way — see `ambi/providers/google.py` for the normalized ↔ provider-native translation pattern.

## Trust model

This is real-execution software running real tools on your machine. The
defaults are personal-use defaults, not multi-tenant production defaults.

- **`run_command`** controls *binaries*, not arguments. Allowlisting `git`
  permits `git push --force`. Tighten by argv validators if your threat
  model needs it.
- **MCP servers** (including Hippocamp) run as subprocesses inheriting your
  shell environment and credentials. Treat each MCP server you wire in as
  trusted code.
- **Skills** are advisory, not authoritative. They describe *how* to use
  capabilities; they do not enforce *what is allowed*. Enforcement lives
  in the tool layer.
- **Telegram transport** defaults to `TELEGRAM_ALLOWED_USER_IDS` empty,
  which means "allow everyone." Set it to your numeric user ID before
  exposing the bot publicly.
- **SenseGate** is a best-effort LLM judge, not a guarantee. It catches
  classes of action-hallucination — it does not catch arbitrary lies.

## Hippocamp companion

[Hippocamp](https://github.com/leosoft-company/hippocamp) is the sibling
project: a portable, persistent memory store accessed over MCP. The
integration ships in `ambi/integrations/hippocamp.py` as a worked example
of the generic MCP wrapping.

```bash
pip install "ambi-core[hippocamp]"
```

Then in `.env`:

```
AMBI_USE_HIPPOCAMP=1
```

`hippocamp-mcp` lives on PATH after install; you only need
`HIPPOCAMP_CMD` if you run from a venv that isn't activated.

## Tests

```bash
uv run pytest -m "not smoke"   # unit suite — no network
uv run pytest -m smoke         # live smoke against real Gemini (needs GEMINI_API_KEY)
uv run pytest                  # both
```

## Project layout

```
ambi/
  loop.py             Agent and chat() loop
  types.py            Message/Block/ToolDef/CompletionResult
  provider.py         LLMProvider Protocol
  providers/google.py Gemini adapter
  tool.py             Tool, ToolRegistry, ToolKind
  skills.py           Skill discovery, catalog, load_skill tool
  sensegate.py        Action verifier (LLM judge + audit log)
  scheduler.py        TaskStore, Scheduler, schedule()/list/cancel tools
  store.py            SqliteStore (session persistence)
  run_command.py      Allowlisted external commands
  mcp.py              McpServer + mcp_tools() wrapper
  integrations/       hippocamp/ — worked MCP example
  transports/         telegram/ — Telegram polling adapter
examples/
  repl.py             Terminal REPL
  telegram_bot.py     Telegram bot (with scheduler + Hippocamp)
  skills/             Demo skills
tests/                pytest, asyncio-mode auto
```

## License

[Apache 2.0](LICENSE)

Authored by [Prageeth Charith](https://github.com/prageethcs), maintained
under [Leosoft](https://github.com/leosoft-company).
