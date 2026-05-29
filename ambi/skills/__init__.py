"""Skills — progressive disclosure via a catalog in the system prompt.

A skill is a markdown file with YAML frontmatter (`name`, `description`) plus
a body. The catalog (one line per skill) is inlined in the agent's system
prompt; bodies are fetched on demand via the built-in `load_skill` tool.

## Authoring conventions: skills are advisory, not authoritative

Skills describe *how and when* to use capabilities. They do NOT enforce
*what is allowed* — that belongs to the code that runs (tool handlers,
policies like CommandPolicy). Skills are just prose, so:

  - Tool calls succeed or fail deterministically; prose is hope.
  - Two sources of truth (skill + policy) drift and produce confusing
    failures: skill says "X is allowed", policy rejects X → user is
    confused; or vice versa, the model ignores a prose prohibition
    under task pressure.

In practice:

  ✅  "Prefer `rg` over `find` when searching."          (workflow)
  ✅  "Don't `git push --force`, even though git works." (social policy)
  ✅  "For commits, use `git log --oneline`."            (idiomatic usage)
  ❌  "Allowed commands: ls, cat, grep, ..."             (duplicates policy)
  ❌  "You may not call X."                              (skill can't enforce)

If a skill needs to surface live policy, point the model at the tool's
own description — tool descriptions can carry runtime-injected truth
(see `make_run_command_tool` for an example).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..tool import Tool
from ..types import ToolDef


@dataclass
class SkillDef:
    name: str
    description: str
    body: str
    filename: str = ""


class SkillRegistry:
    """Instance-scoped registry of skill definitions."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDef] = {}

    @classmethod
    def from_dir(cls, path: str | Path) -> "SkillRegistry":
        return cls.from_dirs(path)

    @classmethod
    def from_dirs(cls, *paths: str | Path) -> "SkillRegistry":
        """Load skills from multiple directories in order.

        Two layouts are supported in each directory:

          1. Skill package:   <name>/SKILL.md  (preferred, can colocate tools)
          2. Flat skill file: <name>.md         (legacy, no tools)

        Later directories shadow earlier ones — same skill name wins later.
        That's how user skills in ~/.ambi/skills/ override the bundled
        defaults shipped in ambi/skills/.
        """
        reg = cls()
        for path in paths:
            skills_path = Path(path)
            if not skills_path.is_dir():
                continue

            # Skill packages: <name>/SKILL.md
            for entry in sorted(skills_path.iterdir()):
                if not entry.is_dir():
                    continue
                skill_md = entry / "SKILL.md"
                if skill_md.exists():
                    skill = _parse_skill_file(skill_md)
                    if skill is not None:
                        # Use the directory name as the skill name if frontmatter
                        # didn't override it.
                        reg._skills[skill.name] = skill

            # Legacy flat .md files at the top level.
            for md_file in sorted(skills_path.glob("*.md")):
                skill = _parse_skill_file(md_file)
                if skill is not None:
                    reg._skills[skill.name] = skill
        return reg

    @classmethod
    def bundled_dir(cls) -> Path:
        """Directory of `.md` skills shipped with the ambi-core package."""
        return Path(__file__).parent

    def register(self, skill: SkillDef) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDef | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills.keys())

    def catalog(self) -> str:
        """Return the catalog block for injection into the system prompt."""
        if not self._skills:
            return ""
        return "\n".join(
            f"- {s.name}: {s.description}"
            for s in sorted(self._skills.values(), key=lambda s: s.name)
        )


def _parse_skill_file(path: Path) -> SkillDef | None:
    text = path.read_text().strip()
    if not text:
        return None

    name = path.stem
    description = ""
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                fm = {}
            name = fm.get("name", path.stem)
            description = fm.get("description", "")
            body = parts[2].strip()

    if not body:
        return None

    return SkillDef(name=name, description=description, body=body, filename=path.name)


_LOAD_SKILL_DESCRIPTION = (
    "Load the full instructions for a named skill. Call this when a user "
    "request matches a skill in the catalog and you need its detailed "
    "guidance before acting. The catalog (skill names + descriptions) is "
    "in your system prompt; pass the matching name here."
)


def make_load_skill_tool(registry: SkillRegistry) -> Tool:
    """Build the `load_skill(name)` tool bound to a SkillRegistry."""

    async def handler(args: dict) -> str:
        name = (args.get("name") or "").strip()
        if not name:
            return "Error: name is required."
        skill = registry.get(name)
        if skill is None:
            available = ", ".join(registry.names()) or "(none)"
            return f"Error: Unknown skill '{name}'. Available: {available}"
        return skill.body

    return Tool(
        definition=ToolDef(
            name="load_skill",
            description=_LOAD_SKILL_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name from the catalog",
                    },
                },
                "required": ["name"],
            },
        ),
        handler=handler,
    )


_CATALOG_PREAMBLE = (
    "SKILL CATALOG: Below is the list of skills available to you. Each entry "
    "is a domain-specific instruction set. When a user request matches a "
    "skill, call load_skill(name) to read its full instructions before "
    "acting. Don't load skills speculatively — only when you've decided one "
    "is needed."
)


def register_bundled_skill_tools(tools) -> None:
    """Walk ambi/skills/* for skill packages that ship a `tools.py` and
    call their `register(tool_registry)` entry point.

    Each bundled skill self-decides whether to wire any tools (typically
    by checking an env var like OBSIDIAN_VAULT).
    """
    import importlib

    bundled_dir = SkillRegistry.bundled_dir()
    for entry in sorted(bundled_dir.iterdir()):
        if not entry.is_dir():
            continue
        tools_module = entry / "tools.py"
        if not tools_module.exists():
            continue
        try:
            mod = importlib.import_module(f"ambi.skills.{entry.name}.tools")
        except Exception as e:
            import sys
            print(
                f"warning: failed to import ambi.skills.{entry.name}.tools ({e})",
                file=sys.stderr,
            )
            continue
        register = getattr(mod, "register", None)
        if callable(register):
            register(tools)


def assemble_system(base: str, registry: SkillRegistry | None) -> str:
    """Append the skill catalog block to the user-supplied system prompt."""
    if registry is None:
        return base
    catalog = registry.catalog()
    if not catalog:
        return base
    return f"{base}\n\n{_CATALOG_PREAMBLE}\n\n{catalog}"
