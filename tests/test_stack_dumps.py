# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import re
from collections import Counter
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# The directory holding tests/, put on the child's PYTHONPATH so that it can import the plugin.
_ROOT = Path(__file__).parents[1]

# Every line the child writes to stderr: a [stuck-test] report or a line of a faulthandler dump.
# Two writers interleaving on stderr break at least one line out of these shapes.
_REPORT_LINE = re.compile(
    r"\[stuck-test\] worker gw\d+ \(pid \d+\) has been running \S+ for \d+\.\d min"
    r"|Timeout \([\d:.]+\)!"
    r"|(Current thread|Thread) 0x[0-9a-f]+ \(most recent call first\):"
    r'|  File ".+", line \d+ in \S+'
    r"|  <no Python frame>"
    r"|"
)


def test_stuck_tests_are_reported_whole(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two copies sleep through ~5 intervals at once on two workers, so their dumps fire together;
    # each must repeat and reach stderr whole, right after the line naming its test.
    pytester.makepyfile(
        """
        import time

        import pytest


        @pytest.mark.parametrize("_copy", range(2))
        def test_sleeps_through_several_dumps(_copy):
            time.sleep(1.0)
        """
    )
    monkeypatch.setenv("PYTHONPATH", str(_ROOT))
    child = pytester.runpytest_subprocess(
        "-p", "tests.config.stack_dumps", "-n", "2", "--dump-stacks-after=0.2", timeout=60
    )
    child.assert_outcomes(passed=2)
    lines = child.stderr.lines
    reports = [i for i, line in enumerate(lines) if line.startswith("[stuck-test]")]
    workers = Counter(lines[i].split()[2] for i in reports)
    assert workers.keys() == {"gw0", "gw1"}, child.stderr.str()
    assert min(workers.values()) >= 2, child.stderr.str()
    assert all(lines[i + 1].startswith("Timeout (") for i in reports), child.stderr.str()
    assert all(_REPORT_LINE.fullmatch(line) for line in lines), child.stderr.str()
