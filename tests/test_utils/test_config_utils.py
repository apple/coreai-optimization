# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging

from coreai_opt._utils.config_utils import get_last_matching_spec


def test_no_match():
    assert get_last_matching_spec(["a"], {"b": 1}) == (None, False)


def test_single_match():
    assert get_last_matching_spec(["a"], {"a": 42, "b": 99}) == (42, True)


def test_wildcard_fallback():
    assert get_last_matching_spec(["a"], {"*": 7}) == (7, True)


def test_specific_match_beats_wildcard():
    assert get_last_matching_spec(["a"], {"a": 1, "*": 99}) == (1, True)


def test_explicit_none_value_is_found():
    value, found = get_last_matching_spec(["a"], {"a": None})
    assert found is True
    assert value is None


def test_last_key_wins_on_multiple_matches():
    # spec_dict order determines precedence, not identifier order
    assert get_last_matching_spec(["a", "b"], {"a": 1, "b": 2}) == (2, True)


def test_last_key_wins_respects_spec_dict_order_not_identifier_order():
    # "b" comes after "a" in spec_dict, so it wins even though "a" is listed first in identifiers
    assert get_last_matching_spec(["b", "a"], {"a": 1, "b": 2}) == (2, True)


def test_warning_emitted_on_multiple_matches(caplog):
    with caplog.at_level(logging.WARNING, logger="coreai_opt._utils.config_utils"):
        get_last_matching_spec(["a", "b"], {"a": 1, "b": 2})
    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    # identifiers and matched keys both appear in the message
    assert "a" in msg and "b" in msg


def test_no_warning_on_single_match(caplog):
    with caplog.at_level(logging.WARNING, logger="coreai_opt._utils.config_utils"):
        get_last_matching_spec(["a"], {"a": 1, "b": 2})
    assert len(caplog.records) == 0


def test_integer_identifiers():
    assert get_last_matching_spec([0, 1], {0: "x", 1: "y", 2: "z"}) == ("y", True)
