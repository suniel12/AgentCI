# The world-file format (v1)

A **world file** is CIAgent's freeze of a failing run's tool traffic: for each
tool the agent called, the arguments it was called with and the response it
got back. Replaying against a world serves those frozen responses instead of
hitting real backends, **fail-closed** — an unmatched call is a recorded miss,
never a guess. This is the one artifact in CIAgent with no standard
equivalent, so it is documented here for anyone who wants to produce or
consume one.

This is documentation, not a standards campaign. Consume it if it's useful.
Machine-readable schema (JSON Schema 2020-12):
[`src/ciagent/schemas/world-file.v1.json`](../src/ciagent/schemas/world-file.v1.json),
which ships with the package — `from ciagent.world import world_file_schema`
returns it, no network fetch.

## Shape

```json
{
  "world_schema": 1,
  "name": "refund-flow",
  "agent": "support-router",
  "frozen_from": {"source": "stage", "id": "...", "envelope_mode": "simulated"},
  "tools": {
    "lookup_invoice": {
      "fixtures": [
        {"match": {"customer_email": "redacted-1@example.com"},
         "response": "Invoice INV-1: $49, paid."}
      ]
    },
    "process_refund": {
      "sequence": true,
      "fixtures": [
        {"match": {"invoice_id": "INV-1"}, "response": "refund initiated"},
        {"match": {"invoice_id": "INV-1"}, "response": "error: already in progress"}
      ]
    }
  }
}
```

## Fields

Only `world_schema` and `tools` are required — that is exactly what the loader
enforces, so a minimal hand-authored world is valid. Everything else is
optional and defaulted on load.

| Field | Where | Meaning |
|---|---|---|
| `world_schema` | top | Format version. **Const 1.** |
| `tools` | top | Map of tool function name → its fixtures. |
| `name`, `agent` | top | Labels for reports/panels. |
| `frozen_from` | top | Provenance of the source (stage id or golden path). |
| `mutated_from` | top | Present only on derived worlds (`world mutate`): operator, tools, payload id, source hash. |
| `gaps` | top | Calls recorded without a result at freeze time; they always miss on replay. |
| `tools[t].fixtures` | tool | Ordered fixtures (required). |
| `tools[t].sequence` | tool | FIFO consumption for state transitions (default false = reusable). |
| `tools[t].suggested_ignore` | tool | Freeze-time mutable-field hints, informational. |
| `fixtures[i].match` | fixture | The frozen call arguments (required). |
| `fixtures[i].response` | fixture | The frozen result, served verbatim (any JSON type; absent = null). |
| `fixtures[i].ignore` | fixture | Match keys that accept any value (mutable fields). |
| `fixtures[i].turn`, `notes` | fixture | Informational. |

## The matching contract

A fixture matches an offered call when:

1. every non-`ignore`d `match` key is present in the offered arguments and
   compares equal after scalar normalization (equal, canonical-JSON-equal, or
   string-equal — this tolerates the framework's type coercion, e.g. `"5"`
   arriving as `5`); and
2. every extra offered key either is `ignore`d or equals that parameter's
   signature default (the framework fills defaults the model omitted).

Otherwise it is a **miss**: the runtime raises and records it. A world never
invents a response and never falls through to the real function. For a
`sequence` tool, fixtures are consumed in order; an exhausted sequence is a
miss.

Reusable (non-sequence) tools must not carry two fixtures with the same
effective match and different responses — loaders reject that as ambiguous.
Use `sequence: true` to express "same call, different result over time."

## Redaction

Worlds frozen from staged entries inherit capture-time redaction, and the
placeholder map is envelope-consistent (a value that appeared in both a user
turn and a tool argument gets the same placeholder, so an agent that echoes
its input still matches). Injected mutation payloads are **not** redacted:
they are authored data, and scrubbing them would neuter an injection gate.

## Compatibility

Within `world_schema: 1`, unknown keys are ignored, so additive fields are
backward-compatible. A version bump (`world_schema: 2`) is a **hard
incompatibility**: a v1 loader rejects it outright rather than reading it
forward. Producers targeting broad compatibility should stay within v1 and
add only optional keys.
