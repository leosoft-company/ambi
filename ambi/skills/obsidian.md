---
name: obsidian
description: Read, search, write, or delete notes in the user's Obsidian vault (PARA-organized).
---

The Obsidian vault is the user's second brain — a directory tree of
markdown files organised by **PARA**:

- **Projects/** — active commitments with a deadline or clear outcome
- **Areas/** — ongoing responsibilities (Work, Family, Health, Finances…)
- **Resources/** — reference material, learning, things to revisit
- **Archive/** — completed projects, retired areas, anything inactive
- **Inbox/** — capture buffer; everything new lands here and gets sorted later

Ground every claim about vault content in a tool call. Never invent note
contents or paths.

## Default save → Inbox

When the user asks you to save, jot, note, or capture something, write to
**Inbox** unless they explicitly name a different folder. Don't try to be
clever about filing on first capture — the user will sort it themselves
during their PARA review. Calling `obsidian_save` without a `folder` arg
lands in Inbox automatically; the tool description confirms the default.

## When to specify a different folder

Only file directly into a PARA bucket when the user's intent is clear:
- "Add this to my Postgres migration project" → `folder: "Projects/Postgres migration"`
- "Save this under my Health area" → `folder: "Areas/Health"`
- "Archive this old project note" → move to `Archive/<original-folder>`

For anything else, prefer Inbox.

## Typical flow

- **Search first**: `obsidian_search({"query": "..."})` — full-text across
  filenames and bodies. Falls back to `obsidian_list` for browsing a folder.
- **Read**: `obsidian_read({"path": "Areas/Work/MOC.md"})` — full content,
  including frontmatter.
- **Save (capture)**: `obsidian_save({"title": "...", "content": "...", "tags": "..."})`
  → lands in Inbox.
- **Save (filed)**: same call with `"folder": "Areas/Health"` when intent is clear.
- **Delete**: destructive — confirm with the user first if the intent is ambiguous.

Always pass the full markdown body in `content` — don't truncate or
summarise the user's input.
