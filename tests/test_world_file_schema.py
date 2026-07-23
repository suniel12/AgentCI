# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Conformance: schemas/world-file.v1.json accepts every world the code
produces (freeze + mutate output) and a minimal hand-authored world, and
rejects malformed ones. This is what makes the published schema real
(enforced) rather than prose. Plan: Plan_docs/standards_adoption.md (S3-S6).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from ciagent.conversation import ConversationEnvelope, ConversationTurn
from ciagent.models import Span, SpanKind, ToolCall, Trace
from ciagent.world import Fixture, ToolWorld, World, freeze_envelope, world_file_schema
from ciagent.world_mutations import mutate_world


@pytest.fixture(scope="module")
def validator():
    schema = world_file_schema()  # the shipped, importable schema
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _env(calls):
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.tool_calls = [ToolCall(tool_name=t, arguments=a, result=r)
                       for t, a, r in calls]
    trace = Trace(agent_name="a", test_name="q", spans=[span])
    trace.metadata["final_output"] = "answer"
    return ConversationEnvelope(
        mode="simulated", agent="a",
        scenario={"name": "s", "spec": {"name": "s", "turns": ["hi"]}},
        turns=[ConversationTurn(turn_index=0, user_message="hi", trace=trace)],
    )


class TestProducedWorldsConform:
    def test_freeze_output_validates(self, validator):
        w = freeze_envelope(_env([("lookup", {"email": "x"}, "found")]))
        validator.validate(w.to_dict())

    def test_sequence_freeze_validates(self, validator):
        w = freeze_envelope(_env([
            ("refund", {"id": "1"}, "initiated"),
            ("refund", {"id": "1"}, "again"),
        ]))
        assert w.tools["refund"].sequence
        validator.validate(w.to_dict())

    def test_mutate_output_validates(self, validator):
        w = freeze_envelope(_env([("lookup", {"email": "x"}, "data")]))
        d, _ = mutate_world(w, "inject", source_path="w.json",
                            payload_id="role-override")
        # derived worlds carry mutated_from
        assert d.to_dict()["mutated_from"]["operator"] == "inject"
        validator.validate(d.to_dict())

    def test_structured_response_validates(self, validator):
        w = freeze_envelope(_env([("lookup", {"e": "x"}, {"records": [1, 2]})]))
        validator.validate(w.to_dict())

    def test_gaps_world_validates(self, validator):
        w = freeze_envelope(_env([
            ("lookup", {"e": "x"}, "ok"),
            ("verify", {"e": "x"}, None),
        ]), allow_gaps=True)
        assert w.gaps
        validator.validate(w.to_dict())


class TestMinimalHandAuthored:
    def test_minimal_world_validates_and_roundtrips(self, validator, tmp_path):
        # S4: the smallest thing World.load accepts must also validate.
        minimal = {"world_schema": 1,
                   "tools": {"t": {"fixtures": [{"match": {}, "response": "r"}]}}}
        validator.validate(minimal)
        p = tmp_path / "m.world.json"
        p.write_text(json.dumps(minimal))
        w = World.load(p)
        # round-trips: to_dict re-validates
        validator.validate(w.to_dict())
        assert w.serve("t", {}) == "r"


class TestRejectsMalformed:
    def test_missing_world_schema_rejected(self, validator):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({"tools": {}})

    def test_wrong_version_rejected(self, validator):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({"world_schema": 2, "tools": {}})

    def test_missing_tools_rejected(self, validator):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({"world_schema": 1})

    def test_fixture_without_match_rejected(self, validator):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({"world_schema": 1,
                                "tools": {"t": {"fixtures": [{"response": "r"}]}}})

    def test_tool_without_fixtures_rejected(self, validator):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({"world_schema": 1, "tools": {"t": {}}})


class TestSchemaAndDocPresent:
    def test_schema_file_referenced_by_doc(self):
        repo = Path(__file__).resolve().parents[1]
        doc = (repo / "docs" / "world-file-schema.md").read_text()
        assert "world-file.v1.json" in doc
        assert "world_schema" in doc
