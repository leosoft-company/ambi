"""Render a static SVG of the ambi REPL panels for the README.

Doesn't actually run the agent — just renders what the UI looks like for
a representative interaction. Re-run when the REPL visuals change.

    uv run python docs/render_demo.py
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


def banner() -> Panel:
    t = Text()
    t.append("ambi ", style="bold magenta")
    t.append("v0.1.0\n", style="dim")
    t.append("Session: 0 messages · ", style="dim")
    t.append("~/.ambi/data/session.db", style="dim cyan")
    t.append("\nCommands: ", style="dim")
    t.append("exit", style="dim bold")
    t.append(" · ", style="dim")
    t.append("history", style="dim bold")
    t.append(" · ", style="dim")
    t.append("audit", style="dim bold")
    return Panel(t, border_style="cyan", padding=(0, 1))


def reply_panel(markdown_body: str, trace: str | None = None) -> Panel:
    body = markdown_body
    if trace:
        body = f"_{trace}_\n\n{markdown_body}"
    return Panel(
        Markdown(body),
        title="[bold magenta]ambi[/bold magenta]",
        title_align="left",
        border_style="magenta",
        padding=(0, 1),
    )


def main() -> None:
    console = Console(record=True, width=92)

    console.print(banner())
    console.print()

    # Turn 1: greeting
    console.print("[bold green]❯[/bold green] hi")
    console.print(reply_panel("hey. what's up?"))
    console.print()

    # Turn 2: tool use + final
    console.print("[bold green]❯[/bold green] what time is it in Tokyo?")
    console.print(
        reply_panel(
            "It's 19:53 JST in Tokyo right now.",
            trace='↳ get_current_time({"timezone": "Asia/Tokyo"})',
        )
    )
    console.print()

    # Turn 3: scheduling
    console.print("[bold green]❯[/bold green] remind me at 5pm tomorrow to review my todos")
    console.print(
        reply_panel(
            "done. fires Sat 17:00 BST. (id: 3a91c4f7)",
            trace='↳ schedule({"prompt": "Remind the user to review their todos.", "run_at": "..."})',
        )
    )
    console.print()

    console.print("[bold green]❯[/bold green] [dim]exit[/dim]")

    console.save_svg(
        "docs/demo.svg",
        title="ambi chat",
        font_aspect_ratio=0.61,
    )


if __name__ == "__main__":
    main()
