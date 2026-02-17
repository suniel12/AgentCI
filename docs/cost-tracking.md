# Cost Tracking

Agent CI automatically tracks the cost of every LLM call.

## Setting Budgets

You can set budgets in `agentci.yaml`:

```yaml
defaults:
  max_cost_usd: 0.10
```

Or for individual tests:

```yaml
tests:
  - name: "expensive_test"
    max_cost_usd: 0.50
```

## failing on Cost

To fail the CI pipeline if cost exceeds a threshold:

```bash
agentci run --fail-on-cost 1.00
```
