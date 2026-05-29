"""Obsidian vault tool tests — operate on a tmp_path vault."""

import pytest

from ambi.integrations.obsidian import (
    VaultError,
    _parse_frontmatter,
    _resolve_under_vault,
    _safe_title,
    make_obsidian_tools,
)


def _tools(tmp_path):
    return {t.definition.name: t for t in make_obsidian_tools(tmp_path)}


# ---------- safety helpers ----------


def test_safe_title_replaces_special_chars():
    assert _safe_title("hello world") == "hello world"
    assert _safe_title("a/b/c") == "a_b_c"
    assert _safe_title("with:colon?and*star") == "with_colon_and_star"


def test_safe_title_empty_raises():
    with pytest.raises(VaultError):
        _safe_title("")
    with pytest.raises(VaultError):
        _safe_title("   ")


def test_resolve_under_vault_rejects_escape(tmp_path):
    with pytest.raises(VaultError):
        _resolve_under_vault(tmp_path, "../escape")
    with pytest.raises(VaultError):
        _resolve_under_vault(tmp_path, "../../etc/passwd")


def test_resolve_under_vault_allows_subpaths(tmp_path):
    assert _resolve_under_vault(tmp_path, "ok") == (tmp_path / "ok").resolve()
    assert _resolve_under_vault(tmp_path, "a/b/c.md") == (tmp_path / "a/b/c.md").resolve()


def test_make_obsidian_tools_rejects_missing_vault(tmp_path):
    with pytest.raises(VaultError):
        make_obsidian_tools(tmp_path / "does-not-exist")


# ---------- frontmatter ----------


def test_parse_frontmatter_with_meta():
    text = "---\ntitle: hello\ntags: [a, b]\n---\nbody here"
    meta, body = _parse_frontmatter(text)
    assert meta["title"] == "hello"
    assert meta["tags"] == ["a", "b"]
    assert body.strip() == "body here"


def test_parse_frontmatter_without_meta():
    text = "no frontmatter"
    meta, body = _parse_frontmatter(text)
    assert meta == {}
    assert body == "no frontmatter"


def test_parse_frontmatter_malformed_yaml_returns_empty():
    text = "---\n: : :\n---\nbody"
    meta, body = _parse_frontmatter(text)
    assert meta == {}
    assert body.strip() == "body"


# ---------- save ----------


async def test_save_defaults_to_inbox(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    result = await save.handler({"title": "test", "content": "Hello world"})
    assert "Saved:" in result
    p = tmp_path / "Inbox" / "test.md"
    assert p.exists(), "default save should land in Inbox/"
    text = p.read_text()
    assert text.startswith("---\n")
    assert "title: test" in text
    assert "created:" in text
    assert "Hello world" in text


async def test_save_custom_default_folder(tmp_path):
    """default_folder parameter overrides Inbox."""
    from ambi.integrations.obsidian import make_obsidian_tools

    tools = {t.definition.name: t for t in make_obsidian_tools(tmp_path, default_folder="Capture")}
    await tools["obsidian_save"].handler({"title": "x", "content": "body"})
    assert (tmp_path / "Capture" / "x.md").exists()


async def test_save_empty_default_folder_lands_at_root(tmp_path):
    """default_folder='' restores old behaviour: save at vault root."""
    from ambi.integrations.obsidian import make_obsidian_tools

    tools = {t.definition.name: t for t in make_obsidian_tools(tmp_path, default_folder="")}
    await tools["obsidian_save"].handler({"title": "x", "content": "body"})
    assert (tmp_path / "x.md").exists()


async def test_save_explicit_folder_overrides_default(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({
        "title": "filed",
        "content": "body",
        "folder": "Areas/Health",
    })
    assert (tmp_path / "Areas/Health/filed.md").exists()
    # And nothing in Inbox.
    assert not (tmp_path / "Inbox" / "filed.md").exists()


async def test_save_in_subfolder_creates_it(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({
        "title": "deep",
        "content": "body",
        "folder": "Areas/Work",
    })
    assert (tmp_path / "Areas/Work/deep.md").exists()


async def test_save_with_tags(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({
        "title": "tagged",
        "content": "x",
        "tags": "ai, research, ml",
    })
    text = (tmp_path / "Inbox" / "tagged.md").read_text()
    assert "tags:" in text
    assert "- ai" in text
    assert "- research" in text


async def test_save_rejects_empty_content(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    result = await save.handler({"title": "x", "content": ""})
    assert "content" in result.lower()


async def test_save_rejects_path_traversal(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    result = await save.handler({
        "title": "evil",
        "content": "x",
        "folder": "../outside",
    })
    assert "escapes" in result or "Error" in result
    # The escape attempt errors out and the file isn't created anywhere
    assert not (tmp_path.parent / "outside").exists()


# ---------- list ----------


async def test_list_empty_vault(tmp_path):
    list_t = _tools(tmp_path)["obsidian_list"]
    result = await list_t.handler({})
    assert "no notes" in result.lower()


async def test_list_finds_notes_recursively(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({"title": "a", "content": "x"})
    await save.handler({"title": "b", "content": "x", "folder": "Areas"})
    list_t = _tools(tmp_path)["obsidian_list"]
    result = await list_t.handler({})
    assert "a.md" in result
    assert "Areas/b.md" in result


async def test_list_filters_by_folder(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    # Inbox is the default, so this lands there.
    await save.handler({"title": "inbox_note", "content": "x"})
    await save.handler({"title": "area_note", "content": "x", "folder": "Areas"})
    list_t = _tools(tmp_path)["obsidian_list"]
    result = await list_t.handler({"folder": "Areas"})
    assert "area_note.md" in result
    assert "inbox_note.md" not in result


# ---------- search ----------


async def test_search_finds_in_body(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({"title": "research", "content": "The mitochondria is the powerhouse."})
    search = _tools(tmp_path)["obsidian_search"]
    result = await search.handler({"query": "mitochondria"})
    assert "research.md" in result
    assert "mitochondria" in result


async def test_search_finds_in_filename(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({"title": "WizardNotes", "content": "unrelated"})
    search = _tools(tmp_path)["obsidian_search"]
    result = await search.handler({"query": "Wizard"})
    assert "WizardNotes.md" in result


async def test_search_no_match_reports_clearly(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({"title": "a", "content": "nothing relevant"})
    search = _tools(tmp_path)["obsidian_search"]
    result = await search.handler({"query": "ZZZUNFINDABLE"})
    assert "no matches" in result.lower()


# ---------- read ----------


async def test_read_returns_full_content(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({"title": "readme", "content": "the body"})
    read = _tools(tmp_path)["obsidian_read"]
    result = await read.handler({"path": "Inbox/readme.md"})
    assert "the body" in result
    assert "title: readme" in result  # frontmatter included


async def test_read_missing_path(tmp_path):
    read = _tools(tmp_path)["obsidian_read"]
    result = await read.handler({"path": "ghost.md"})
    assert "not found" in result


async def test_read_rejects_traversal(tmp_path):
    read = _tools(tmp_path)["obsidian_read"]
    result = await read.handler({"path": "../etc/passwd"})
    assert "escapes" in result or "Error" in result


# ---------- delete ----------


async def test_delete_removes_file(tmp_path):
    save = _tools(tmp_path)["obsidian_save"]
    await save.handler({"title": "doomed", "content": "x"})
    assert (tmp_path / "Inbox" / "doomed.md").exists()
    delete = _tools(tmp_path)["obsidian_delete"]
    result = await delete.handler({"path": "Inbox/doomed.md"})
    assert "Deleted" in result
    assert not (tmp_path / "Inbox" / "doomed.md").exists()


async def test_delete_missing_is_noop(tmp_path):
    delete = _tools(tmp_path)["obsidian_delete"]
    result = await delete.handler({"path": "ghost.md"})
    assert "already absent" in result


# ---------- kinds ----------


def test_tool_kinds(tmp_path):
    tools = _tools(tmp_path)
    assert tools["obsidian_save"].kind == "write"
    assert tools["obsidian_delete"].kind == "write"
    assert tools["obsidian_list"].kind == "read"
    assert tools["obsidian_search"].kind == "read"
    assert tools["obsidian_read"].kind == "read"
