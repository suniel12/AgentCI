# Continuous Integration with GitHub Actions

AgentCI is designed to run in your CI/CD pipeline to catch regressions before they hit production.

## Quick Setup

1. Copy the template from `.github/workflows/agentci-template.yml` to your repository's `.github/workflows/` directory.

```bash
mkdir -p .github/workflows
cp path/to/agentci-template.yml .github/workflows/agentci.yml
```

2. Set your API keys in the GitHub Repository Settings -> Security -> Secrets and variables -> Actions.
    - `OPENAI_API_KEY`
    - `ANTHROPIC_API_KEY`

3. The action will now run on every push and pull request.

## Configuration

The default template runs the following command:

```yaml
run: agentci run --ci --fail-on-cost 0.50
```

- `--ci`: Ensures the process exits with code 1 if any test fails or errors.
- `--fail-on-cost 0.50`: Fails the build if the total cost of the test suite exceeds $0.50.

## Artifacts

The workflow is configured to upload an HTML report of the test results. You can find this in the "Artifacts" section of your GitHub Action run summary.

```yaml
- name: Upload test report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: agentci-report
    path: agentci-report.html
```
