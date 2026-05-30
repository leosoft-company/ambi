# Security model

ambi is real-execution software: it runs real tools on your machine with your
credentials. The defaults are **personal-use** defaults — a single trusted
operator on their own box — not multi-tenant production defaults. This document
describes the threat model, the layered defenses, how to configure them, and
the residual gaps we know about and have *not* solved. We'd rather name the
holes than imply they don't exist.

## Threat model

The primary threat is **prompt injection via tool output**. ambi reads
untrusted external content all the time — fetched pages, file contents, email
bodies, and the responses of any MCP server you wire in. A compromised or
malicious source can embed text crafted to hijack the agent ("ignore previous
instructions, you are now…", or an encoded equivalent).

A successful hijack matters only in proportion to what the agent can then *do*.
The realistic kill chain we defend against:

> poisoned tool output → agent induced to read a secret (`cat .env`) → agent
> induced to exfiltrate it (`git push` to an attacker remote, or an outbound
> message)

Out of scope: a malicious *operator* (you can already run anything yourself), a
compromised host OS, supply-chain attacks on dependencies, and side channels.
MCP servers and the Python tool code you install are **trusted** — treat each
MCP server you wire in as trusted code running with your environment.

## Defense layers

Two independent kinds of control, because the first kind is probabilistic and
the second is structural.

### 1. Reduce the odds of a hijack (prompt-level, soft)

- **Trust envelope.** Every tool result sent to the model is wrapped in
  `<tool_output trust="data">…</tool_output>` (built in `ambi/types.py`,
  applied in the `Agent` context view in `ambi/loop.py`). The system prompt
  (`ambi/cli/system.md`) frames envelope contents as *data to reason about,
  never commands to obey* — including decoded/encoded payloads.
- **Sanitizer.** `sanitize_tool_output()` neutralizes common obfuscation before
  the model sees it: strips zero-width/invisible chars and Unicode "tag"
  smuggling, flags bidi overrides, folds compatibility forms (fullwidth,
  ligatures) via NFKC, and redacts plaintext markers ("ignore previous
  instructions", forged `</tool_output>`/`<system>` tags). A `sanitized="true"`
  attribute on the envelope warns the model when a marker was stripped.
- **Hardened secondary LLMs.** The SenseGate verifier and the compaction
  summarizer both read attacker-controllable tool output. Their inputs are
  sanitized + enveloped and their prompts instruct them to never obey embedded
  instructions — so a result can't talk the integrity checker into
  rubber-stamping a lie, or poison a summary that persists into future turns.

These lower the probability of an opportunistic hijack. They do **not** stop a
determined, targeted adversary — see *Residual gaps*.

### 2. Cap the blast radius if a hijack succeeds (structural, hard)

Enforcement lives in the tool layer, not in prompts. The **Warden**
(`ambi/warden.py`) authorizes every tool call *before* it runs: policies are
evaluated in order, first non-`allow` verdict wins, every decision is audited.
The default stack wired by `ambi/cli/build.py`:

| Policy | Effect |
|---|---|
| `ArgvValidatorPolicy` | Denies destructive `run_command` shapes (`git push --force`, `rm -rf /`, …). Matches case-insensitively by substring **and** ordered token subsequence, so injected flags / extra whitespace can't evade it. |
| `UrlAllowlistPolicy` | **Hard-denies** egress to any host not in the allowlist (default `github.com`, `gitlab.com`, `bitbucket.org`). Scans argv for `scheme://host` and scp-style `user@host:path`. This is the primary exfil cut. |
| `RequireConfirmationPolicy` | Forces human approval for `git push` / `git remote add|set-url`. Returns `require_confirmation`, gated by the agent's `confirm` hook. |
| `CostCeilingPolicy` | Denies further tool calls once daily LLM spend exceeds the ceiling. |

`run_command` itself (`ambi/run_command.py`, `CommandPolicy`) adds three gates
independent of the Warden:

- **argv[0] allowlist** — only allowlisted binaries run. Default is read-mostly
  (`ls, pwd, cat, head, tail, wc, git, date, echo, grep`). `find` is
  deliberately excluded (its `-exec` launches arbitrary binaries).
- **forbidden args** — blocks argument-level escapes (`-exec`, `-execdir`,
  `-delete`, …) regardless of allowlist, so re-adding `find` can't reopen the
  hole.
- **secret-path denylist** — refuses to read sensitive paths (`*/.env`,
  `*/.ssh/*`, `id_rsa*`, `*/.aws/*`, `~/.config/gh`, `~/.ambi/.env`, …),
  checked against each argv token's raw, `~`-expanded, and resolved forms.
  Committed templates (`.env.example`, `.sample`, …) are exempt.

### Fail-closed confirmation

`require_confirmation` is a real gate, not a suggestion. The agent loop calls an
optional async `confirm` hook; **if no confirmer is wired (e.g. the headless
Telegram daemon), or it declines or raises, the call is not executed.** `ambi
chat` wires an interactive y/N confirmer. The daemon does not — so outbound
actions fail closed there by default.

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `AMBI_RUN_COMMAND_ALLOW` | `ls,pwd,cat,head,tail,wc,git,date,echo,grep` | argv[0] allowlist for `run_command`. |
| `AMBI_ALLOWED_GIT_HOSTS` | `github.com,gitlab.com,bitbucket.org` | Extra hosts `run_command` may push/clone to (added to the egress allowlist). |
| `AMBI_DAILY_COST_USD` | `1.00` | Daily LLM spend ceiling before tools are denied. |
| `TELEGRAM_ALLOWED_USER_IDS` | *(empty = allow everyone)* | Numeric user IDs allowed to talk to the bot. **Set this before exposing the bot.** |

Programmatic builds can pass any `Warden(policies=[…])` and an `Agent(confirm=…)`
hook. Custom policies satisfy the `Policy` protocol (`evaluate(ctx) ->
PolicyDecision`); custom `CommandPolicy` fields tune the allowlist, forbidden
args, secret paths, and cwd jail.

## Residual gaps (known, unsolved)

We list these so they're not mistaken for covered ground:

- **Prompt-level defenses are probabilistic.** The envelope + sanitizer reduce
  opportunistic injection; a targeted adversary using base64/hex/rot13 content
  encodings or cross-script homoglyphs can still get a payload past the
  sanitizer. The structural controls (Warden, allowlists, confirmation) are
  what actually hold — design assuming the model *can* be hijacked.
- **The egress allowlist only inspects `run_command` argv.** If you wire a
  `WebFetch`/HTTP tool or MCP tools that make outbound calls, they need their
  own egress/confirmation policy — `RequireConfirmationPolicy(tools={…})`
  handles the named-tool case.
- **Denylists are bypassable by equivalent spellings.** `ArgvValidatorPolicy`,
  forbidden-args, and the secret-path list are defense-in-depth, not proofs.
  An allowlisted interpreter (if you add `python`, `bash`, etc.) defeats the
  whole model — keep the allowlist tight; that's the real gate.
- **MCP servers and installed tool code are trusted.** They run as subprocesses
  inheriting your environment and credentials. Vet what you wire in.
- **SenseGate is best-effort.** It catches classes of action-hallucination; it
  is not a guarantee and is not an authorization control.

## Reporting

This is alpha software. To report a vulnerability, open an issue or contact the
maintainer at prageeth.cs@gmail.com.
