"""Regression tests for the stack-based Markdown hierarchy parser.

These tests pin the exact bug the user reported: parent sections like
"(2) General Securities Principal" being dropped, leaving only the leaf
"(A) Requirement" in chunk paths and merging unrelated subsections.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_server"))

from core.nodes import _insert_into_tree, _render_tree
from core.structure import parse_markdown_hierarchy, split_large_section


def test_full_path_preserved_for_nested_headings():
    """`### (A) Requirement` under `## (2) ...` keeps both ancestors."""
    md = (
        "## (2) General Securities Principal\n"
        "### (A) Requirement\n"
        "Each member shall designate a Principal.\n"
    )
    groups = parse_markdown_hierarchy(md)
    assert len(groups) == 1
    assert groups[0]["path"] == [
        "(2) General Securities Principal",
        "(A) Requirement",
    ]


def test_sibling_subsections_with_same_name_stay_distinct():
    """Two `### (A) Requirement` blocks under different parents must NOT merge."""
    md = (
        "## (1) General Securities Representative\n"
        "### (A) Requirement\n"
        "rep-requirement\n"
        "\n"
        "## (2) General Securities Principal\n"
        "### (A) Requirement\n"
        "principal-requirement\n"
    )
    groups = parse_markdown_hierarchy(md)
    paths = [g["path"] for g in groups]
    assert paths == [
        ["(1) General Securities Representative", "(A) Requirement"],
        ["(2) General Securities Principal", "(A) Requirement"],
    ]
    # And the content is bound to the correct parent.
    assert "rep-requirement" in groups[0]["content"][0]
    assert "principal-requirement" in groups[1]["content"][0]


def test_shallower_heading_clears_deeper_levels():
    """An H2 after an H4 must reset H3 and H4 from the live stack."""
    md = (
        "## A\n"
        "### B\n"
        "#### C\n"
        "deep\n"
        "\n"
        "## D\n"
        "shallow\n"
    )
    groups = parse_markdown_hierarchy(md)
    assert groups[0]["path"] == ["A", "B", "C"]
    # 'D' is a fresh top-level section — must NOT inherit B/C.
    assert groups[1]["path"] == ["D"]


def test_split_large_section_copies_path_onto_every_piece():
    """Oversized groups split internally — every chunk keeps the same path."""
    section = {
        "path": ["(2) General Securities Principal", "(A) Requirement"],
        "content": ["x" * 4000, "y" * 4000, "z" * 4000],
    }
    pieces = split_large_section(section, max_chars=6000)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece["path"] == [
            "(2) General Securities Principal",
            "(A) Requirement",
        ]


def test_aggregate_renders_as_nested_markdown():
    """Tree built from path tuples renders with correct heading depths."""
    root = {"summaries": [], "children": {}}
    _insert_into_tree(
        root,
        ["(2) General Securities Principal", "(A) Requirement"],
        "- bullet about requirement",
        index=0,
    )
    _insert_into_tree(
        root,
        ["(2) General Securities Principal", "(B) Number"],
        "- bullet about number",
        index=1,
    )

    lines: list[str] = []
    _render_tree(root, depth=1, lines=lines)
    rendered = "\n".join(lines)

    assert "## (2) General Securities Principal" in rendered
    assert "### (A) Requirement" in rendered
    assert "### (B) Number" in rendered
    # Parent must precede its children in the output.
    assert rendered.index("## (2)") < rendered.index("### (A)")
