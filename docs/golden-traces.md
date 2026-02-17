# Golden Traces

Golden traces are the "known good" execution path for your agent.

## Recording a Golden Trace

To record a new golden trace:

```bash
agentci record my_test_case
```

This will run your agent live and save the trace to `golden/my_test_case.golden.json`.

## Diffing

When you run `agentci run`, the new trace is compared against the golden trace. Differences are highlighted:

- **Tools Changed**: Different tools were called.
- **Args Changed**: Arguments to tools changed.
- **Cost Spike**: The cost increased significantly.
