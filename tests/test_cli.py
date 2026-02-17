"""
Tests for the CLI.
"""
from click.testing import CliRunner
from agentci.cli import cli

def test_init_command():
    runner = CliRunner()
    result = runner.invoke(cli, ['init'])
    assert result.exit_code == 0
