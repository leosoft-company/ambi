# ambi-core architecture

## Package layout

```
ambi-core/
├── ambi/
│   ├── types.py         # Message, Block (Text/ToolUse/ToolResult), CompletionResult, ToolDef
│   ├── tool.py          # Tool, ToolRegistry  (register, defs, invoke)
│   ├── skills.py        # SkillDef, SkillRegistry, make_load_skill_tool, assemble_system
│   ├── provider.py      # LLMProvider Protocol  (complete, stream)
│   ├── loop.py          # Agent (stateful: messages accumulate)
│   ├── env.py           # load_env, require_env
│   └── providers/
│       └── google.py    # GoogleProvider — google-genai adapter
├── examples/
│   ├── repl.py          # interactive chat REPL
│   └── skills/time.md   # demo skill
├── tests/
│   ├── mock_provider.py # scripted CompletionResult player
│   ├── test_skills.py   # 18 tests
│   ├── test_tool.py     # 3 tests
│   ├── test_loop.py     # 8 tests (offline, with MockProvider)
│   └── test_smoke_google.py  # 3 tests, hit real Gemini (pytest -m smoke)
└── pyproject.toml       # hatchling build, pytest-asyncio
```

## Component graph

```
                  ┌──────────────────────────────┐
                  │  caller code (e.g. repl.py)  │
                  └──────────────┬───────────────┘
                                 │ agent.chat(user_input) -> str
                                 ▼
       ┌──────────────────────────────────────────────────────┐
       │                       Agent                          │
       │  - messages: list[Message]   (persistent session)    │
       │  - system: str               (base + skill catalog)  │
       │  - chat() loop:                                      │
       │      while turns < max:                              │
       │        result = provider.complete(messages, ...)     │
       │        append assistant blocks                       │
       │        if stop != tool_use: return final text        │
       │        invoke all tool_use blocks in parallel        │
       │        append tool_results as user msg               │
       └──┬──────────────────────┬────────────────────┬───────┘
          │ tools.invoke(name)   │ provider.complete  │ skills.catalog
          ▼                      ▼                    ▼ (at init)
   ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
   │ ToolRegistry │      │ LLMProvider  │      │  SkillRegistry   │
   │  register    │      │  (Protocol)  │      │  from_dir(.md)   │
   │  defs        │      └──────┬───────┘      │  get / catalog   │
   │  invoke ────┐│             │              └──────┬───────────┘
   └─────────────┘│             │                     │
                  │             ▼                     │ load_skill tool
                  │      ┌──────────────────┐         │ auto-registered
                  │      │ GoogleProvider   │◄────────┘ at Agent init
                  │      │  _to_gemini_*    │
                  │      │  _from_gemini_*  │
                  │      │  _schema_to_*    │
                  │      └────────┬─────────┘
                  │               │ google-genai SDK
                  │               ▼
                  │      ┌──────────────────┐
                  │      │   Gemini API     │
                  │      └──────────────────┘
                  ▼
           ┌──────────────┐
           │ user tools + │
           │ load_skill   │
           └──────────────┘
```

## Runtime sequence — one `chat()` call with a tool round-trip

```
caller          Agent             Provider           ToolRegistry        Skill body
  │               │                  │                    │                  │
  │ chat("...")   │                  │                    │                  │
  ├──────────────►│                  │                    │                  │
  │               │ append user msg  │                    │                  │
  │               │ complete(msgs, tools, system)         │                  │
  │               ├─────────────────►│                    │                  │
  │               │                  │  HTTP → Gemini     │                  │
  │               │◄─────────────────┤  CompletionResult  │                  │
  │               │                  │  (stop=tool_use)   │                  │
  │               │ append assistant │                    │                  │
  │               │ gather(invoke(c) for c in tool_uses)  │                  │
  │               ├──────────────────────────────────────►│                  │
  │               │◄──────────────────────────────────────┤ ToolResultBlocks │
  │               │ append as user msg (tool_results)     │                  │
  │               │ complete(...)    │                    │                  │
  │               ├─────────────────►│                    │                  │
  │               │◄─────────────────┤  stop=end_turn     │                  │
  │               │ append assistant │                    │                  │
  │  final text   │                  │                    │                  │
  │◄──────────────┤                  │                    │                  │
```

If the tool happens to be `load_skill(name)`, the registry returns the skill body and it flows back as the `ToolResultBlock` content — same path, no special case.

## Provider seam

The `LLMProvider` Protocol is the only contract:

```
complete(messages, tools, system, max_tokens, **kwargs) -> CompletionResult
```

`messages` and `tools` are in our normalized shape (Anthropic-influenced — `ToolResultBlock` lives inside `user` messages). Each adapter translates in/out:

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
- No `tool_use_id` — matched by name; we stash `_tool_name` on `ToolResultBlock` and use it on translation
- `function_response.response` must be a dict — wrap `{"result": ...}` / `{"error": ...}`
- System prompt goes in `config`, not the message stream
- JSON Schema sanitized (strip `$schema`, `additionalProperties`, etc.) before passing as `gt.Schema`
- `FinishReason` compared by enum name string for SDK forward-compat

## Skills — progressive disclosure

1. `SkillRegistry.from_dir("skills")` parses `*.md` frontmatter at startup (cheap).
2. `Agent.__init__(skills=registry)`:
   - splices `SKILL CATALOG:\n- name: description\n...` into the system prompt via `assemble_system()`
   - registers the built-in `load_skill(name)` tool, bound to the registry
3. Model sees the catalog every turn, picks a skill, calls `load_skill("pdf")` → tool returns body → body enters history as a `ToolResultBlock` → model acts on it.
4. No body elision; no "loaded" tracking. Bloat is left to (future) conversation compaction, which is a separate concern any continuous-session agent has.

## Stateful session

`Agent` is stateful — `self.messages` accumulates across `chat()` calls. There is no `reset()`. To start fresh, construct a new `Agent`. The provider receives the full `self.messages` list every turn.

## Concurrency

Tool calls within a single turn run in parallel via `asyncio.gather`. The order of `ToolResultBlock`s in the resulting user message preserves the order the model emitted them in (zip-by-position), so `tool_use_id`s line up correctly.
