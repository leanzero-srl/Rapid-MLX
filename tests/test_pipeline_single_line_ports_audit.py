# SPDX-License-Identifier: Apache-2.0
"""No single-engine fix is missed by the pipeline line without a reviewed reason (goose Q-144).

The pipeline line (tags ``lz-pipeline-qwen4.*``) branched from the single
engine's line (tags ``v*-lz.*``) at v0.14.3-lz.2.  Every fix made on the
single line after that point is absent here unless someone ports it — lz.6's
tool-message tail was missed that way until goose Q-143, and the Q-144 audit
found lz.7 and lz.8 missing too.

This test lists every non-merge commit reachable from the newest single-line
tag but not from HEAD (nor from the upstream release both lines sit on), and
fails on any that ``pipeline_single_line_ports.json`` does not review.  A
review names the pipeline commits that carry the fix (each must be in HEAD's
history), or says why the pipeline server has no such code path.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("pipeline_single_line_ports.json")
STATUSES = {"ported", "equivalent", "n/a"}
SINGLE_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)-lz\.(\d+)")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture(scope="module")
def history() -> None:
    try:
        inside = _git("rev-parse", "--is-inside-work-tree")
    except (OSError, subprocess.CalledProcessError) as absent:
        pytest.skip(
            f"not a git checkout of the fork, the audit reads its history: {absent}"
        )
    assert inside == "true"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _newest_single_tag() -> str:
    tags = [t for t in _git("tag", "-l", "v*-lz.*").split() if SINGLE_TAG.fullmatch(t)]
    assert tags, (
        "no v*-lz.* tag in this checkout, so the single engine's line cannot be "
        "seen: `git fetch --tags`"
    )
    return max(
        tags, key=lambda t: tuple(int(n) for n in SINGLE_TAG.fullmatch(t).groups())
    )


def test_every_single_line_fix_is_reviewed_for_the_pipeline(history, manifest):
    newest = _newest_single_tag()
    missing = _git(
        "rev-list",
        "--no-merges",
        "--reverse",
        newest,
        "^HEAD",
        f"^{manifest['upstream_base']}",
    ).split()
    unreviewed = [sha for sha in missing if sha not in manifest["commits"]]
    assert not unreviewed, (
        f"single-line commits up to {newest} that this pipeline line neither contains "
        "nor reviews — port each, or record it in tests/pipeline_single_line_ports.json "
        "with the reason the pipeline server has no such code path:\n"
        + "\n".join(_git("log", "-1", "--format=%H %s", sha) for sha in unreviewed)
    )
    assert manifest["reviewed_through"] == newest, (
        f"the review stops at {manifest['reviewed_through']}, the single line is at {newest}"
    )


def test_every_review_is_checkable(history, manifest):
    for sha, review in manifest["commits"].items():
        assert review["status"] in STATUSES, (sha, review["status"])
        assert review.get("reason", "").strip(), f"{sha}: a review says why"
        assert _git("cat-file", "-t", sha) == "commit", sha
        if review["status"] == "n/a":
            assert not review.get("pipeline"), f"{sha}: n/a names no pipeline commit"
            continue
        assert review.get("pipeline"), (
            f"{sha}: {review['status']} names its pipeline commits"
        )
        for carried in review["pipeline"]:
            ancestor = subprocess.run(
                ["git", "-C", str(REPO), "merge-base", "--is-ancestor", carried, "HEAD"]
            ).returncode
            assert ancestor == 0, (
                f"{sha} is '{review['status']}' by {carried}, not in HEAD"
            )
