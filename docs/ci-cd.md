# CI/CD Setup

Agent CI is designed to run in GitHub Actions, GitLab CI, and other CI providers.

## GitHub Actions

Copy the template from `.github/workflows/agentci-template.yml` to your repo.

```yaml
jobs:
  agent-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install agentci
      - name: Run Tests
        run: agentci run --ci
```
