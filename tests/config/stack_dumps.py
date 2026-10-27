# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pytest plugin that reports stuck tests without failing them.

``--dump-stacks-after=SECONDS`` (default 900, 0 disables) makes faulthandler dump the thread
stacks of a test's process for every SECONDS the test keeps running. Under pytest-xdist, the
controller prints each dump whole, after a line naming the worker and the test.
"""

import faulthandler
import os
import threading
import time
from pathlib import Path

import pytest
from xdist.workermanage import WorkerController

# In a process that runs tests: the --dump-stacks-after interval, the fd faulthandler dumps to,
# and under xdist the fd of the file naming the running test for the controller. None when the
# dumps are off and in the controller. Set by pytest_sessionstart.
_stack_dump: tuple[float, int, int | None] | None = None

# In the xdist controller: the directory each worker writes <worker>.dump and <worker>.test to.
_stack_dump_dir: str | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --dump-stacks-after."""
    parser.addoption(
        "--dump-stacks-after",
        type=float,
        # Above the slowest healthy tests, which take about 10.5 minutes on a loaded CI machine,
        # so passing runs print no dumps.
        default=900.0,
        metavar="SECONDS",
        help="every SECONDS a test keeps running, dump all thread stacks and name the test "
        "without failing it (default: 900; 0 disables)",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Pick where this process writes stack dumps; in the xdist controller, start the printer."""
    global _stack_dump, _stack_dump_dir
    interval = session.config.getoption("dump_stacks_after")
    if interval <= 0:
        return
    # xdist registers "dsession" only in a controller that distributes tests. Its trylast
    # pytest_sessionstart starts the workers after this one, so the directory exists when
    # pytest_configure_node hands it to them.
    if session.config.pluginmanager.has_plugin("dsession"):
        # pytest's per-run temp directory, which xdist also reads; pytest prunes old runs, even
        # ones CI killed mid-hang.
        dump_dir = session.config._tmp_path_factory.mktemp("stack-dumps")
        _stack_dump_dir = str(dump_dir)
        printer = threading.Thread(
            target=_print_stack_dumps,
            # Poll several times per interval, so each dump is printed after its own line.
            args=(dump_dir, min(1.0, interval / 4)),
            name="stack-dump-printer",
            daemon=True,
        )
        printer.start()
        return
    workerinput = getattr(session.config, "workerinput", None)
    if workerinput is None:
        # A single process has no other dumps to interleave with, so faulthandler writes to
        # stderr directly.
        _stack_dump = (interval, os.dup(2), None)
        return
    path = os.path.join(workerinput["stack_dump_dir"], workerinput["workerid"])
    _stack_dump = (
        interval,
        os.open(f"{path}.dump", os.O_WRONLY | os.O_CREAT | os.O_APPEND),
        os.open(f"{path}.test", os.O_WRONLY | os.O_CREAT),
    )


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: WorkerController) -> None:
    """Tell an xdist worker where to write its stack dumps."""
    node.workerinput["stack_dump_dir"] = _stack_dump_dir


def pytest_runtest_logstart(nodeid: str) -> None:
    """Schedule the repeating stack dump of the test that is starting."""
    if _stack_dump is None:
        return
    interval, dump_fd, test_fd = _stack_dump
    if test_fd is not None:
        # The controller reads this to name the test in the line it prints before each dump.
        test = f"{time.time()} {os.getpid()} {nodeid}".encode()
        os.pwrite(test_fd, test, 0)
        os.ftruncate(test_fd, len(test))
    # Dumps from a C thread, so it still fires when the hang never releases the GIL.
    faulthandler.dump_traceback_later(interval, repeat=True, file=dump_fd)


def pytest_runtest_logfinish() -> None:
    """Cancel the stack dump scheduled by pytest_runtest_logstart."""
    faulthandler.cancel_dump_traceback_later()


def _print_stack_dumps(dump_dir: Path, poll: float) -> None:
    """Copy each worker's new stack dumps to stderr whole, after a line naming the test.

    faulthandler writes one dump in many small writes, so dumps from workers sharing stderr
    interleave. Each worker writes to its own file instead, and this one thread copies a file's
    new bytes in a single write once the file stops growing.
    """
    stderr_fd = os.dup(2)
    printed: dict[Path, int] = {}  # bytes of each .dump file already copied to stderr
    sizes: dict[Path, int] = {}  # size of each .dump file at the previous poll
    while True:
        time.sleep(poll)
        for path in sorted(dump_dir.glob("*.dump")):
            size = path.stat().st_size
            done = printed.get(path, 0)
            previous, sizes[path] = sizes.get(path), size
            # Skip a file with nothing new, or one faulthandler may still be writing to.
            if size == done or size != previous:
                continue
            with path.open("rb") as dump_file:
                dump_file.seek(done)
                dump = dump_file.read(size - done)
            printed[path] = size
            started, pid, nodeid = path.with_suffix(".test").read_text().split(" ", 2)
            # Start on a fresh line: pytest's progress dots on stdout can leave a log that merges
            # stdout and stderr mid-line.
            report = (
                f"\n[stuck-test] worker {path.stem} (pid {pid}) has been running {nodeid} "
                f"for {(time.time() - float(started)) / 60:.1f} min\n"
            )
            os.write(stderr_fd, report.encode() + dump)
