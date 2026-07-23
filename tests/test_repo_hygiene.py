"""Repo hygiene: git history must stay free of AI co-author trailers.

The full history was rewritten on 2026-07-23 to remove them. This test keeps
the suite red if one ever reappears. Two other layers enforce the same rule:
the commit-hygiene job in .github/workflows/ci.yml (full-history scan) and
the hooks in .githooks/ (block at commit and push time).

Note: in a shallow CI checkout this test only sees the fetched commits; the
commit-hygiene CI job checks the complete history with fetch-depth 0.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAILER_PATTERN = r"co-authored-by:.*(claude|anthropic)"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


def test_no_ai_coauthor_trailer_in_history():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("not running from a git checkout")

    result = _git(
        "log",
        "--all",
        "-i",
        "-E",
        f"--grep={TRAILER_PATTERN}",
        "--format=%h %s",
    )
    assert result.returncode == 0, result.stderr
    offenders = result.stdout.strip()
    assert offenders == "", (
        "AI co-author trailer found in commit message(s):\n"
        f"{offenders}\n"
        "Amend or rebase to drop the Co-Authored-By trailer."
    )
