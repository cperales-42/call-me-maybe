"""Real-time terminal dashboard for constrained generation."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading

from pydantic import BaseModel, PrivateAttr

from .generator import GenerationEvent, generate_prompt
from .grammar import Grammar
from .models import FunctionCallResult, ModelProtocol
from .tokenizer import SimpleTokenizer

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_CLEAR_SCREEN = "\x1b[2J\x1b[H"

_Rgb = tuple[int, int, int]

_ORANGE_BORDER: _Rgb = (230, 126, 34)
_RED_BORDER: _Rgb = (231, 76, 60)
_GREEN_BORDER: _Rgb = (46, 204, 113)
_RESULTS_BORDER: _Rgb = (52, 152, 219)

_KEY_COLOR: _Rgb = (86, 182, 194)
_STRING_COLOR: _Rgb = (152, 195, 121)
_NUMBER_COLOR: _Rgb = (229, 192, 123)
_LITERAL_COLOR: _Rgb = (198, 120, 221)
_PUNCT_COLOR: _Rgb = (120, 130, 140)
_DIM_COLOR: _Rgb = (110, 118, 129)
_ACCENT_COLOR: _Rgb = (229, 160, 80)
_PROMPT_COLOR: _Rgb = (198, 208, 220)
_ALLOWED_COLOR: _Rgb = (152, 195, 121)
_WRITING_COLOR: _Rgb = (229, 192, 123)
_COMPLETE_COLOR: _Rgb = (152, 195, 121)

_PANEL_HEIGHT = 4
_PROMPT_HEIGHT = 4
_ALLOWED_HEIGHT = 2
_RESULTS_MIN_HEIGHT = 3
_CONSTRAINT_VISIBLE_SECONDS = 0.5
_MIN_SIDEBAR = 18
_MAX_SIDEBAR = 34

# Shared by every visualizer instance created during one process, so results
# from earlier prompts remain visible while later prompts are generated.
_RESULTS_HISTORY: list[str] = []


def _fg(color: _Rgb) -> str:
    """Return the ANSI sequence for a 24-bit foreground color.

    Args:
        color: Red, green, and blue components in the 0-255 range.

    Returns:
        The escape sequence that activates the color.
    """
    red, green, blue = color
    return f"\x1b[38;2;{red};{green};{blue}m"


def _bg(color: _Rgb) -> str:
    """Return the ANSI sequence for a 24-bit background color.

    Args:
        color: Red, green, and blue components in the 0-255 range.

    Returns:
        The escape sequence that activates the color.
    """
    red, green, blue = color
    return f"\x1b[48;2;{red};{green};{blue}m"


def _style(
    text: str,
    *,
    fg: _Rgb | None = None,
    bg: _Rgb | None = None,
    bold: bool = False,
    color: bool,
) -> str:
    """Apply optional terminal styling to text.

    Args:
        text: Text to style.
        fg: Optional foreground color.
        bg: Optional background color.
        bold: Whether to use bold text.
        color: Whether ANSI styling is enabled.

    Returns:
        The styled text, or the original text when styling is disabled.
    """
    if not color:
        return text
    prefix = _BOLD if bold else ""
    if fg is not None:
        prefix += _fg(fg)
    if bg is not None:
        prefix += _bg(bg)
    return f"{prefix}{text}{_RESET}"


def _stdout_is_interactive() -> bool:
    """Return whether stdout is attached to a terminal.

    Returns:
        ``True`` for an interactive terminal, otherwise ``False``.
    """
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _colors_enabled() -> bool:
    """Return whether ANSI color output should be used.

    Returns:
        ``True`` when colors are appropriate for the current terminal.
    """
    if not _stdout_is_interactive():
        return False
    if sys.platform == "win32" and "ANSICON" not in os.environ:
        return False
    return os.environ.get("NO_COLOR") is None and os.environ.get(
        "TERM"
    ) != "dumb"


def _supports_unicode() -> bool:
    """Return whether stdout can encode the dashboard glyphs.

    Returns:
        ``True`` when the terminal encoding accepts the visual glyphs.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "▌│…·█".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _terminal_size() -> tuple[int, int]:
    """Return a safe terminal width and height.

    Returns:
        A ``(columns, lines)`` pair with usable minimum values.
    """
    size = shutil.get_terminal_size(fallback=(100, 30))
    return max(40, size.columns), max(12, size.lines)


def _fit(text: str, width: int, *, unicode_ok: bool = True) -> str:
    """Fit text into a fixed number of terminal columns.

    Args:
        text: Text to fit.
        width: Maximum visible width.
        unicode_ok: Whether an ellipsis glyph can be used.

    Returns:
        Text no longer than ``width`` characters.
    """
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    marker = "…" if unicode_ok else "..."
    if width <= len(marker):
        return text[:width]
    return text[: width - len(marker)] + marker


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text into lines of at most ``width`` characters.

    Args:
        text: Text to wrap.
        width: Maximum line width.

    Returns:
        Wrapped lines; an empty string produces one empty line.
    """
    if width <= 0:
        return [""]
    if text == "":
        return [""]
    return [
        text[index:index + width]
        for index in range(0, len(text), width)
    ]


def _visible_piece(piece: str, *, unicode_ok: bool) -> str:
    """Render a token piece while keeping whitespace and controls visible.

    Args:
        piece: Raw vocabulary token piece.
        unicode_ok: Whether visual Unicode glyphs can be used.

    Returns:
        A printable representation of the piece.
    """
    if not unicode_ok:
        replacements = {
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
            " ": "_",
        }
        return "".join(
            replacements.get(character, character)
            if character.isprintable() or character in replacements
            else f"\\x{ord(character):02x}"
            for character in piece
        )
    replacements = {
        "\n": "⏎",
        "\r": "␍",
        "\t": "⇥",
        " ": "·",
    }
    return "".join(
        replacements.get(character, character)
        if character.isprintable() or character in replacements
        else f"\\x{ord(character):02x}"
        for character in piece
    )


def _last_nonspace(parts: list[str]) -> str:
    """Return the last non-whitespace character in accumulated output.

    Args:
        parts: Previously appended output fragments.

    Returns:
        The last non-whitespace character, or an empty string.
    """
    for part in reversed(parts):
        for character in reversed(part):
            if not character.isspace():
                return character
    return ""


def _pretty_partial_json(raw: str) -> list[str]:
    """Indent a JSON document that may still be incomplete.

    The generated text is structurally valid at every step, but it may end
    inside a string, after an opening brace, or in the middle of a number.
    The formatter therefore scans characters rather than calling ``json``.

    Args:
        raw: Accumulated generated JSON text.

    Returns:
        Pretty-printed lines, preserving incomplete trailing content.
    """
    output: list[str] = []
    indent = 0
    in_string = False
    escaped = False

    for index, character in enumerate(raw):
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            output.append(character)
        elif character in "{[":
            output.append(character)
            opener = character
            following = raw[index + 1] if index + 1 < len(raw) else ""
            closing = "}" if opener == "{" else "]"
            if following == closing:
                continue
            indent += 1
            output.append("\n" + "  " * indent)
        elif character in "}]":
            expected = "{" if character == "}" else "["
            if _last_nonspace(output) == expected:
                output.append(character)
            else:
                indent = max(0, indent - 1)
                output.append("\n" + "  " * indent + character)
        elif character == ",":
            output.append(",\n" + "  " * indent)
        elif character == ":":
            output.append(": ")
        elif character in " \t\r\n":
            continue
        else:
            output.append(character)

    text = "".join(output)
    return text.splitlines() if text else [""]


def _highlight_json(text: str, *, color: bool) -> str:
    """Apply lightweight JSON syntax highlighting.

    Args:
        text: One raw JSON line.
        color: Whether ANSI styling is enabled.

    Returns:
        The line with keys, strings, numbers, and punctuation colored.
    """
    if not color or text == "":
        return text

    result: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character == '"':
            end = index + 1
            escaped = False
            while end < length:
                current = text[end]
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    end += 1
                    break
                end += 1
            token = text[index:end]
            lookahead = end
            while lookahead < length and text[lookahead] == " ":
                lookahead += 1
            is_key = lookahead < length and text[lookahead] == ":"
            color_code = _KEY_COLOR if is_key else _STRING_COLOR
            result.append(_style(token, fg=color_code, color=color))
            index = end
            continue

        if character in "{}[],:":
            result.append(_style(character, fg=_PUNCT_COLOR, color=color))
            index += 1
            continue

        if character.isspace():
            result.append(character)
            index += 1
            continue

        end = index
        while (
            end < length
            and not text[end].isspace()
            and text[end] not in '{}[],:"'
        ):
            end += 1
        token = text[index:end]
        token_color = (
            _LITERAL_COLOR
            if token in {"true", "false", "null"}
            else _NUMBER_COLOR
        )
        result.append(_style(token, fg=token_color, color=color))
        index = end

    return "".join(result)


def _plain_line(
    text: str,
    width: int,
    *,
    fg: _Rgb | None = None,
    bold: bool = False,
    color: bool,
    unicode_ok: bool,
) -> str:
    """Create a padded plain-text column line.

    Args:
        text: Line content.
        width: Target visible width.
        fg: Optional foreground color.
        bold: Whether to use bold text.
        color: Whether ANSI styling is enabled.
        unicode_ok: Whether truncated text may use an ellipsis glyph.

    Returns:
        A line padded to ``width`` visible columns.
    """
    padded = _fit(text, width, unicode_ok=unicode_ok).ljust(width)
    return _style(padded, fg=fg, bold=bold, color=color)


def _box_lines(
    title: str,
    text_rows: list[str],
    width: int,
    height: int,
    *,
    border: _Rgb,
    color: bool,
    unicode_ok: bool,
) -> list[str]:
    """Draw a rectangle around already-prepared content rows.

    The interior keeps the terminal's normal background so only the border
    carries the panel color.

    Args:
        title: Panel title embedded in the top border.
        text_rows: Content rows, already wrapped to the panel width.
        width: Panel width.
        height: Panel height including both border rows.
        border: Border color.
        color: Whether ANSI styling is enabled.
        unicode_ok: Whether visual Unicode box glyphs can be used.

    Returns:
        Padded and styled rectangle lines.
    """
    if unicode_ok:
        top_left, top_right = "┌", "┐"
        bottom_left, bottom_right = "└", "┘"
        horizontal, vertical = "─", "│"
    else:
        top_left = top_right = bottom_left = bottom_right = "+"
        horizontal, vertical = "-", "|"

    safe_title = _fit(title, max(0, width - 5), unicode_ok=unicode_ok)
    prefix = f"{top_left}{horizontal} {safe_title} "
    remaining = max(0, width - len(prefix) - 1)
    top = prefix + horizontal * remaining + top_right
    bottom = bottom_left + horizontal * max(0, width - 2) + bottom_right

    inner_width = max(1, width - 4)
    content_height = max(0, height - 2)
    visible = text_rows[:content_height]
    visible.extend("" for _ in range(content_height - len(visible)))

    lines = [
        _style(
            top.ljust(width)[:width],
            fg=border,
            bold=True,
            color=color,
        )
    ]
    for row in visible:
        pad = _fit(row, inner_width, unicode_ok=unicode_ok).ljust(
            inner_width
        )
        lines.append(
            _style(vertical, fg=border, bold=True, color=color)
            + " "
            + pad
            + " "
            + _style(vertical, fg=border, bold=True, color=color)
        )
    lines.append(
        _style(
            bottom.ljust(width)[:width],
            fg=border,
            bold=True,
            color=color,
        )
    )
    return lines[:height]


def _panel_lines(
    title: str,
    piece: str,
    width: int,
    height: int,
    *,
    border: _Rgb,
    color: bool,
    unicode_ok: bool,
    token_piece: bool = True,
    placeholder: str | None = None,
) -> list[str]:
    """Build a rectangle that shows one piece or prompt.

    Args:
        title: Panel title embedded in the top border.
        piece: Text to display inside the panel.
        width: Panel width.
        height: Panel height including both border rows.
        border: Border color.
        color: Whether ANSI styling is enabled.
        unicode_ok: Whether visual Unicode box glyphs can be used.
        token_piece: Whether whitespace should be rendered visibly as part
            of a token piece.
        placeholder: Text to show when ``piece`` is empty. When omitted, a
            waiting message is used.

    Returns:
        Padded and styled rectangle lines.
    """
    inner_width = max(1, width - 4)
    if piece and token_piece:
        display = _visible_piece(piece, unicode_ok=unicode_ok)
    elif piece:
        display = piece.replace("\r", " ").replace("\n", " ")
    elif placeholder is not None:
        display = placeholder
    else:
        display = "waiting…" if unicode_ok else "waiting..."
    wrapped = _wrap(display, inner_width)
    content_height = max(0, height - 2)
    visible = wrapped[:content_height]
    if len(wrapped) > len(visible) and visible:
        visible[-1] = _fit(
            visible[-1] + ("…" if unicode_ok else "..."),
            inner_width,
            unicode_ok=unicode_ok,
        )
    return _box_lines(
        title,
        visible,
        width,
        height,
        border=border,
        color=color,
        unicode_ok=unicode_ok,
    )


def _results_lines(
    results: list[str],
    width: int,
    height: int,
    *,
    border: _Rgb,
    color: bool,
    unicode_ok: bool,
) -> list[str]:
    """Build a scrolling array of completed compact JSON results.

    Args:
        results: Compact JSON strings for completed prompts.
        width: Panel width.
        height: Panel height including both border rows.
        border: Border color.
        color: Whether ANSI styling is enabled.
        unicode_ok: Whether visual Unicode glyphs can be used.

    Returns:
        Padded and styled rectangle lines.
    """
    inner_width = max(1, width - 4)
    content_height = max(0, height - 2)
    if not results:
        entries = ["[]"]
    elif content_height == 0:
        entries = []
    else:
        entries = ["["]
        for index, result in enumerate(results):
            suffix = "," if index < len(results) - 1 else ""
            for part in _wrap(result + suffix, max(1, inner_width - 2)):
                entries.append(f"  {part}")
        entries.append("]")
        if len(entries) > content_height:
            marker = "…" if unicode_ok else "..."
            if content_height == 1:
                entries = [marker]
            else:
                entries = [marker] + entries[-(content_height - 1):]
    title = f"results ({len(results)})" if results else "results"
    return _box_lines(
        title,
        entries,
        width,
        height,
        border=border,
        color=color,
        unicode_ok=unicode_ok,
    )


def _allowed_lines(
    allowed: int | None,
    width: int,
    *,
    color: bool,
    unicode_ok: bool,
) -> list[str]:
    """Build the allowed-token summary shown below the two panels.

    Args:
        allowed: Number of tokens allowed in the latest step, if any.
        width: Sidebar width.
        color: Whether ANSI styling is enabled.
        unicode_ok: Whether visual Unicode glyphs can be used.

    Returns:
        Two padded lines describing the allowed token count.
    """
    if allowed is None:
        value = "waiting"
    else:
        noun = "option" if allowed == 1 else "options"
        value = f"{allowed} {noun}"
    return [
        _plain_line(
            " allowed",
            width,
            fg=_PROMPT_COLOR,
            bold=True,
            color=color,
            unicode_ok=unicode_ok,
        ),
        _plain_line(
            f" {value}",
            width,
            fg=_ALLOWED_COLOR,
            color=color,
            unicode_ok=unicode_ok,
        ),
    ]


def _justify(
    left: str,
    right: str,
    width: int,
    *,
    left_color: _Rgb,
    right_color: _Rgb,
    color: bool,
    unicode_ok: bool,
) -> str:
    """Render left- and right-aligned text within one row.

    Args:
        left: Left-hand text.
        right: Right-hand text.
        width: Available terminal columns.
        left_color: Color for the left-hand text.
        right_color: Color for the right-hand text.
        color: Whether ANSI styling is enabled.
        unicode_ok: Whether truncated text may use an ellipsis glyph.

    Returns:
        A styled row with both labels aligned.
    """
    if width <= 0:
        return ""
    if len(right) >= width:
        right = _fit(right, width, unicode_ok=unicode_ok)
        return _style(right, fg=right_color, bold=True, color=color)
    left = _fit(
        left,
        max(0, width - len(right) - 1),
        unicode_ok=unicode_ok,
    )
    gap = " " * max(1, width - len(left) - len(right))
    return (
        _style(left, fg=left_color, bold=True, color=color)
        + gap
        + _style(right, fg=right_color, bold=True, color=color)
    )


class CLIVisualizer(BaseModel):
    """Render a live terminal view of constrained generation.

    The dashboard keeps the generated JSON in the foreground, with a prompt
    panel across the top and a sidebar that shows the latest model choice,
    the last constraint-forced piece, and the number of allowed tokens.
    """

    show_scores: bool = False

    _prompt: str = PrivateAttr(default="")
    _stream: list[str] = PrivateAttr(default_factory=list)
    _model_piece: str = PrivateAttr(default="")
    _constraint_piece: str = PrivateAttr(default="")
    _last_forced: str = PrivateAttr(default="")
    _allowed: int | None = PrivateAttr(default=None)
    _results: list[str] = PrivateAttr(default_factory=list)
    _final_lines: list[str] = PrivateAttr(default_factory=list)
    _complete: bool = PrivateAttr(default=False)
    _live: bool = PrivateAttr(default=False)
    _color: bool = PrivateAttr(default=False)
    _unicode: bool = PrivateAttr(default=True)
    _constraint_timer: threading.Timer | None = PrivateAttr(default=None)
    _constraint_generation: int = PrivateAttr(default=0)
    _render_lock: threading.RLock = PrivateAttr(
        default_factory=threading.RLock
    )

    def bind_results(self, results: list[str]) -> None:
        """Share one completed-results history across visualizers.

        Args:
            results: Mutable list where completed results are appended.
        """
        self._results = results

    def start(self, prompt: str) -> None:
        """Prepare the dashboard for a prompt.

        Args:
            prompt: Natural-language request being processed.
        """
        self._prompt = prompt
        self._stream = []
        self._model_piece = ""
        self._last_forced = ""
        self._allowed = None
        self._final_lines = []
        self._complete = False
        self._live = _stdout_is_interactive()
        self._color = _colors_enabled()
        self._unicode = _supports_unicode()
        self._constraint_piece = "…" if self._unicode else "..."
        self._constraint_generation += 1
        self._cancel_constraint_timer()

        if self._live:
            sys.stdout.write(_HIDE_CURSOR)
            self._render()

    def on_event(self, event: GenerationEvent) -> None:
        """Record and display one token-selection event.

        Args:
            event: Event emitted by the constrained generator.
        """
        with self._render_lock:
            self._stream.append(event.piece)
            self._allowed = event.allowed_tokens
            if event.source == "model":
                self._model_piece = event.piece
            else:
                self._last_forced = event.piece
                self._constraint_piece = event.piece
                self._schedule_constraint_blank()

            if self._live:
                self._render()

    def _schedule_constraint_blank(self) -> None:
        """Hide the constraint after a short readable delay."""
        if not self._live:
            return
        self._cancel_constraint_timer()
        self._constraint_generation += 1
        timer = threading.Timer(
            _CONSTRAINT_VISIBLE_SECONDS,
            self._blank_constraint,
            args=(self._constraint_generation,),
        )
        timer.daemon = True
        self._constraint_timer = timer
        timer.start()

    def _cancel_constraint_timer(self) -> None:
        """Cancel a pending constraint-blanking timer."""
        timer = self._constraint_timer
        if timer is not None:
            timer.cancel()
            self._constraint_timer = None

    def _blank_constraint(self, generation: int) -> None:
        """Clear the constraint panel once its visible delay elapses."""
        with self._render_lock:
            if generation != self._constraint_generation:
                return
            if (
                not self._live
                or self._complete
                or not self._constraint_piece
            ):
                return
            self._constraint_piece = ""
            self._render()

    def finish(self, result: FunctionCallResult) -> None:
        """Show the completed JSON document.

        Args:
            result: Completed function-call result.
        """
        self._complete = True
        self._cancel_constraint_timer()
        if self._last_forced:
            self._constraint_piece = self._last_forced
        payload = result.model_dump(mode="json")
        document = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        self._final_lines = document.splitlines()
        self._results.append(
            json.dumps(payload, ensure_ascii=False)
        )
        self._render()
        if self._live:
            sys.stdout.write(_SHOW_CURSOR + "\n")
            sys.stdout.flush()

    def abort(self) -> None:
        """Restore the terminal cursor after an interrupted generation."""
        self._complete = True
        self._cancel_constraint_timer()
        if self._live:
            sys.stdout.write(_SHOW_CURSOR + "\n")
            sys.stdout.flush()

    def _render(self) -> None:
        """Redraw the dashboard using the current terminal size."""
        with self._render_lock:
            width, height = _terminal_size()
            lines = self._screen_lines(width, height)
            if self._live:
                sys.stdout.write(_CLEAR_SCREEN)
            sys.stdout.write("\n".join(lines))
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _screen_lines(self, width: int, height: int) -> list[str]:
        """Compose the full dashboard for one terminal size.

        Args:
            width: Terminal width in columns.
            height: Terminal height in lines.

        Returns:
            Dashboard lines ready to write to the terminal.
        """
        header = self._header_lines(width)
        prompt_panel = _panel_lines(
            "prompt",
            self._prompt,
            width,
            _PROMPT_HEIGHT,
            border=_GREEN_BORDER,
            color=self._color,
            unicode_ok=self._unicode,
            token_piece=False,
        )
        overhead = len(header) + len(prompt_panel) + 1
        body_height = max(
            _PANEL_HEIGHT * 2 + _ALLOWED_HEIGHT + _RESULTS_MIN_HEIGHT,
            height - overhead - 1,
        )
        sidebar_width = min(
            _MAX_SIDEBAR,
            max(_MIN_SIDEBAR, width // 3),
        )
        gap = " │ " if self._unicode else " | "
        main_width = max(12, width - sidebar_width - len(gap))

        sidebar = (
            _panel_lines(
                "model",
                self._model_piece,
                sidebar_width,
                _PANEL_HEIGHT,
                border=_ORANGE_BORDER,
                color=self._color,
                unicode_ok=self._unicode,
            )
            + _panel_lines(
                "constraint",
                self._constraint_piece,
                sidebar_width,
                _PANEL_HEIGHT,
                border=_RED_BORDER,
                color=self._color,
                unicode_ok=self._unicode,
                placeholder="",
            )
            + _allowed_lines(
                self._allowed,
                sidebar_width,
                color=self._color,
                unicode_ok=self._unicode,
            )
        )
        sidebar += _results_lines(
            self._results,
            sidebar_width,
            max(_RESULTS_MIN_HEIGHT, body_height - len(sidebar)),
            border=_RESULTS_BORDER,
            color=self._color,
            unicode_ok=self._unicode,
        )
        main = self._main_lines(main_width, body_height)

        rows: list[str] = []
        for index in range(body_height):
            left = (
                sidebar[index]
                if index < len(sidebar)
                else " " * sidebar_width
            )
            right = main[index] if index < len(main) else ""
            rows.append(f"{left}{gap}{right}")
        return header + prompt_panel + [""] + rows

    def _header_lines(self, width: int) -> list[str]:
        """Build the dashboard heading.

        Args:
            width: Terminal width in columns.

        Returns:
            One heading line.
        """
        status = "complete" if self._complete else "generating"
        status_color = _COMPLETE_COLOR if self._complete else _WRITING_COLOR
        heading = (
            " call_me_maybe · constrained decoding"
            if self._unicode
            else " call_me_maybe - constrained decoding"
        )
        title = _justify(
            heading,
            status,
            width,
            left_color=_KEY_COLOR,
            right_color=status_color,
            color=self._color,
            unicode_ok=self._unicode,
        )
        return [title]

    def _main_lines(self, width: int, height: int) -> list[str]:
        """Build the main JSON document column.

        Args:
            width: Main column width.
            height: Available rows.

        Returns:
            A title row, a spacer, and the visible JSON lines.
        """
        status = "complete" if self._complete else "writing"
        status_color = _COMPLETE_COLOR if self._complete else _WRITING_COLOR
        title = _justify(
            " function_call.json",
            status,
            width,
            left_color=_PROMPT_COLOR,
            right_color=status_color,
            color=self._color,
            unicode_ok=self._unicode,
        )

        if self._complete:
            raw_lines = self._final_lines or [""]
        else:
            raw_lines = _pretty_partial_json("".join(self._stream))

        available = max(1, height - 2)
        start = max(0, len(raw_lines) - available)
        visible = raw_lines[start:]
        number_width = max(2, len(str(len(raw_lines))))
        content_width = max(1, width - number_width - 2)
        cursor = "▌" if self._unicode else "_"

        lines = [title, ""]
        for offset, raw_line in enumerate(visible):
            absolute = start + offset + 1
            is_cursor_line = (
                not self._complete and offset == len(visible) - 1
            )
            text_width = content_width - (1 if is_cursor_line else 0)
            text = _fit(
                raw_line,
                max(1, text_width),
                unicode_ok=self._unicode,
            )
            highlighted = _highlight_json(text, color=self._color)
            if is_cursor_line:
                highlighted += _style(
                    cursor,
                    fg=_ACCENT_COLOR,
                    bold=True,
                    color=self._color,
                )
            number = _style(
                f"{absolute:>{number_width}}  ",
                fg=_DIM_COLOR,
                color=self._color,
            )
            lines.append(f"{number}{highlighted}")
        return lines


def visualize_generation(
    model: ModelProtocol,
    grammar: Grammar,
    prompt: str,
    inference_prompt: str,
    max_iter: int,
    *,
    tokenizer: SimpleTokenizer | None = None,
) -> FunctionCallResult:
    """Generate a call while showing its token-by-token progress.

    Args:
        model: Model used to produce next-token logits.
        grammar: Grammar that constrains generation.
        prompt: Original user request stored in the result.
        inference_prompt: Expanded prompt supplied to the model.
        max_iter: Maximum number of tokens to generate.
        tokenizer: Optional shared tokenizer.

    Returns:
        The completed and validated function-call result.
    """
    visualizer = CLIVisualizer()
    visualizer.bind_results(_RESULTS_HISTORY)
    try:
        visualizer.start(prompt)
        result = generate_prompt(
            model,
            grammar,
            prompt,
            inference_prompt,
            max_iter,
            tokenizer=tokenizer,
            observer=visualizer.on_event,
        )
    except BaseException:
        visualizer.abort()
        raise
    visualizer.finish(result)
    return result
