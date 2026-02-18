#!/bin/bash
set -e

# Simulate GitHub Actions Environment
echo "🚀 Simulating CI/CD Pipeline..."

# 1. Setup (Simulate 'Checkout' and 'Setup Python')
# We assume we are in the root of the repo
REPO_ROOT=$(pwd)
TEMP_DIR="ci_simulation_$(date +%s)"

echo "📂 Creating temporary workspace: $TEMP_DIR"
mkdir "$TEMP_DIR"
cd "$TEMP_DIR"

# 2. Install (Simulate 'pip install agentci')
# in a real CI environment, we'd install from PyPI or git.
# Here we install the local version in editable mode to test the CURRENT code.
echo "📦 Installing AgentCI from local source..."
pip install -e "$REPO_ROOT" --quiet

# 3. Init (Simulate user creating a project)
echo "⚙️  Initializing new AgentCI project..."
agentci init

# 4. Run (The actual CI step)
echo "🏃 Running AgentCI tests in CI mode..."
# We need to make sure we have a valid agent to run.
# The init command creates 'agentci.yaml' which points to 'agent:run_agent'.
# We need to create a dummy agent file for it to import.

echo "   ...Creating dummy agent.py..."
cat <<EOF > agent.py
def run_agent(input_text):
    # Retrieve the active trace context if any, or just return mock string
    # The default assertions expect cost/steps, so we need to simulate activity?
    # Actually checking agentci.yaml template:
    # assertions: cost_under 0.01
    # This should pass even if we do nothing, as cost is 0.
    return "Hello world response"
EOF

# Run the command defined in the workflow
# env vars are inherited from the shell (so .env needs to be loaded by the user or this script)
# We'll assert exit code 0
if agentci run --ci --fail-on-cost 0.50; then
    echo "✅ CI Run Passed!"
else
    echo "❌ CI Run Failed!"
    exit 1
fi

# Cleanup
cd ..
rm -rf "$TEMP_DIR"
echo "🧹 Cleanup complete."
echo "🎉 Verification Successful!"
