# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for ``RequestFilter`` — the per-step filter on computed base
date / time used by the MARS source.

The public YAML surface is the wildcard shorthand on ``date`` / ``time``.
The legacy ``user_date`` / ``user_time`` keys are explicitly rejected.
"""

import pytest

from anemoi.datasets.create.sources.mars.retrieval import RequestFilter


class TestRequestFilterEmpty:

    def test_default_is_empty(self) -> None:
        f = RequestFilter()
        assert f.is_empty
        assert f.keep("20200101", "0000")
        assert f.keep("19000101", "1800")

    def test_plain_request_no_filter(self) -> None:
        request = {"class": "ea", "param": "2t", "step": 0}
        f, cleaned = RequestFilter.extract(request)
        assert f.is_empty
        assert cleaned == request

    def test_literal_date_not_a_filter(self) -> None:
        f, cleaned = RequestFilter.extract({"date": "20200101", "time": 0})
        assert f.is_empty
        assert cleaned == {"date": "20200101", "time": 0}


class TestRequestFilterWildcard:

    def test_wildcard_date_only(self) -> None:
        f, cleaned = RequestFilter.extract({"date": "????-??-01", "param": "2t"})
        assert "date" not in cleaned
        assert cleaned == {"param": "2t"}
        assert f.date is not None
        assert f.time is None
        assert f.keep("20200101", "0000")
        assert f.keep("20200101", "1200")
        assert not f.keep("20200102", "0000")

    def test_wildcard_date_and_time(self) -> None:
        f, cleaned = RequestFilter.extract({"date": "????-??-01", "time": 0, "param": "2t"})
        assert "date" not in cleaned
        assert "time" not in cleaned
        assert cleaned == {"param": "2t"}
        assert f.keep("20200101", "0000")
        assert not f.keep("20200101", "1200")  # time fails
        assert not f.keep("20200102", "0000")  # date fails

    def test_wildcard_date_time_list(self) -> None:
        f, _ = RequestFilter.extract({"date": "????????", "time": [0, 1200]})
        assert f.time == frozenset({"0000", "1200"})
        assert f.keep("20200101", "0000")
        assert f.keep("20200101", "1200")
        assert not f.keep("20200101", "0600")

    def test_wildcard_full_any_date(self) -> None:
        f, _ = RequestFilter.extract({"date": "????????", "time": 0})
        # Matches every well-formed YYYYMMDD.
        assert f.keep("20200101", "0000")
        assert f.keep("19991231", "0000")
        assert not f.keep("20200101", "1200")

    def test_wildcard_date_without_dashes(self) -> None:
        f, _ = RequestFilter.extract({"date": "??????01"})
        assert f.keep("20200101", "0000")
        assert not f.keep("20200102", "0000")


class TestLegacyKeysRejected:

    def test_user_date_rejected(self) -> None:
        with pytest.raises(ValueError, match="user_date"):
            RequestFilter.extract({"user_date": "??????01"})

    def test_user_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="user_time"):
            RequestFilter.extract({"user_time": [0]})

    def test_rejection_mentions_shorthand(self) -> None:
        with pytest.raises(ValueError, match="wildcard"):
            RequestFilter.extract({"user_date": "??????01"})


class TestRequestFilterImmutable:

    def test_extract_does_not_mutate_input(self) -> None:
        request = {"date": "????-??-01", "time": [0], "param": "2t"}
        snapshot = dict(request)
        RequestFilter.extract(request)
        assert request == snapshot
