"""
Generate agentci_spec.schema.json from the Pydantic AgentCISpec model.

Usage:
    python -m agentci.schema.generate_schema
"""

import json
from pathlib import Path

from agentci.schema.spec_models import AgentCISpec


def generate() -> None:
    schema = AgentCISpec.model_json_schema()
    out = Path(__file__).parent / "agentci_spec.schema.json"
    out.write_text(json.dumps(schema, indent=2))
    print(f"Schema written to {out}")


if __name__ == "__main__":
    generate()
