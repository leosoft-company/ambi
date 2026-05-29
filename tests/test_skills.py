from pathlib import Path

import pytest

from ambi.skills import (
    SkillDef,
    SkillRegistry,
    _parse_skill_file,
    assemble_system,
    make_load_skill_tool,
)


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content)
    return p


# ---------- _parse_skill_file ----------


def test_parse_with_frontmatter(tmp_path):
    p = _write(
        tmp_path,
        "pdf.md",
        "---\nname: pdf\ndescription: Handle PDFs\n---\nUse the pdf tool to ...",
    )
    s = _parse_skill_file(p)
    assert s == SkillDef(
        name="pdf",
        description="Handle PDFs",
        body="Use the pdf tool to ...",
        filename="pdf.md",
    )


def test_parse_falls_back_to_filename_when_no_frontmatter(tmp_path):
    p = _write(tmp_path, "calendar.md", "Just a body, no frontmatter.")
    s = _parse_skill_file(p)
    assert s is not None
    assert s.name == "calendar"
    assert s.description == ""
    assert s.body == "Just a body, no frontmatter."


def test_parse_returns_none_for_empty_body(tmp_path):
    p = _write(tmp_path, "empty.md", "---\nname: empty\ndescription: x\n---\n   ")
    assert _parse_skill_file(p) is None


def test_parse_returns_none_for_empty_file(tmp_path):
    p = _write(tmp_path, "blank.md", "")
    assert _parse_skill_file(p) is None


def test_parse_malformed_yaml_falls_back(tmp_path):
    p = _write(tmp_path, "bad.md", "---\nname: bad\n: : :\n---\nBody here.")
    s = _parse_skill_file(p)
    assert s is not None
    # Malformed YAML -> empty frontmatter -> name from filename
    assert s.name == "bad"
    assert s.body == "Body here."


# ---------- SkillRegistry ----------


def test_from_dir_loads_all_skills(tmp_path):
    _write(tmp_path, "b.md", "---\nname: b\ndescription: B\n---\nB body")
    _write(tmp_path, "a.md", "---\nname: a\ndescription: A\n---\nA body")
    reg = SkillRegistry.from_dir(tmp_path)
    assert reg.names() == ["a", "b"]


def test_from_dir_missing_returns_empty():
    reg = SkillRegistry.from_dir("/nonexistent/path/xyz")
    assert reg.names() == []
    assert reg.catalog() == ""


def test_from_dir_skips_unparseable(tmp_path):
    _write(tmp_path, "good.md", "---\nname: good\ndescription: G\n---\nbody")
    _write(tmp_path, "empty.md", "")
    reg = SkillRegistry.from_dir(tmp_path)
    assert reg.names() == ["good"]


def test_catalog_format(tmp_path):
    _write(tmp_path, "b.md", "---\nname: b\ndescription: B desc\n---\nbody")
    _write(tmp_path, "a.md", "---\nname: a\ndescription: A desc\n---\nbody")
    reg = SkillRegistry.from_dir(tmp_path)
    assert reg.catalog() == "- a: A desc\n- b: B desc"


def test_get_returns_none_for_unknown():
    reg = SkillRegistry()
    assert reg.get("nope") is None


# ---------- assemble_system ----------


def test_assemble_system_no_registry():
    assert assemble_system("base", None) == "base"


def test_assemble_system_empty_registry():
    assert assemble_system("base", SkillRegistry()) == "base"


def test_assemble_system_appends_catalog(tmp_path):
    _write(tmp_path, "x.md", "---\nname: x\ndescription: X\n---\nbody")
    reg = SkillRegistry.from_dir(tmp_path)
    out = assemble_system("You are helpful.", reg)
    assert out.startswith("You are helpful.\n\nSKILL CATALOG:")
    assert out.endswith("- x: X")


# ---------- load_skill tool ----------


async def test_load_skill_returns_body():
    reg = SkillRegistry()
    reg.register(SkillDef(name="pdf", description="d", body="PDF instructions here"))
    tool = make_load_skill_tool(reg)
    result = await tool.handler({"name": "pdf"})
    assert result == "PDF instructions here"


async def test_load_skill_unknown_lists_available():
    reg = SkillRegistry()
    reg.register(SkillDef(name="a", description="d", body="x"))
    reg.register(SkillDef(name="b", description="d", body="x"))
    tool = make_load_skill_tool(reg)
    result = await tool.handler({"name": "missing"})
    assert "Unknown skill 'missing'" in result
    assert "a, b" in result


async def test_load_skill_missing_name():
    tool = make_load_skill_tool(SkillRegistry())
    result = await tool.handler({})
    assert result == "Error: name is required."


async def test_load_skill_blank_name():
    tool = make_load_skill_tool(SkillRegistry())
    result = await tool.handler({"name": "   "})
    assert result == "Error: name is required."


def test_load_skill_tool_definition_shape():
    tool = make_load_skill_tool(SkillRegistry())
    assert tool.definition.name == "load_skill"
    assert tool.definition.input_schema["required"] == ["name"]
    assert tool.definition.input_schema["properties"]["name"]["type"] == "string"
