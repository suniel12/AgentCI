# Writing Tests

Agent CI tests are defined in `agentci.yaml` or using the Python `@agentci.test` decorator.

## YAML Format

```yaml
name: "my_agent_tests"
agent: "my_module:run_agent"

tests:
  - name: "happy_path"
    input: "Do the thing"
    assertions:
      - type: tool_called
        tool: do_thing
```

## Python Format

```python
import agentci

@agentci.test(input="Do the thing")
def test_happy_path(agent):
    return agent.run("Do the thing")
```
