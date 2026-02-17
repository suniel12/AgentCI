"""
Agent CI Command Line Interface.

Commands:
  agentci init          Scaffold a new test suite
  agentci run           Execute test suite
  agentci run --runs N  Statistical mode (run N times)
  agentci record        Run agent live, save golden trace
  agentci diff          Compare latest run against golden
  agentci report        Generate HTML report from last run
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
@click.version_option()
def cli():
    """Agent CI — Continuous Integration for AI Agents"""
    pass


@cli.command()
def init():
    """Scaffold a new Agent CI test suite."""
    # Creates: agentci.yaml, golden/, demo test case
    pass


from .config import load_config
from .runner import TestRunner
from .models import TestResult

@cli.command()
@click.option('--suite', '-s', default='agentci.yaml', help='Path to test suite YAML')
@click.option('--runs', '-n', default=1, help='Number of runs for statistical mode')
@click.option('--tag', '-t', multiple=True, help='Only run tests with these tags')
@click.option('--diff/--no-diff', default=True, help='Compare against golden traces')
@click.option('--html', type=click.Path(), help='Generate HTML report at this path')
@click.option('--fail-on-cost', type=float, help='Fail if total cost exceeds threshold')
@click.option('--ci', is_flag=True, help='CI mode: exit code 1 on any failure')
def run(suite, runs, tag, diff, html, fail_on_cost, ci):
    """Execute the test suite."""
    console.print(f"[bold blue]Agent CI[/] Running suite: [cyan]{suite}[/]")
    
    try:
        config = load_config(suite)
        runner = TestRunner(config)
        
        # Filter tests by tag if provided
        if tag:
            config.tests = [t for t in config.tests if any(tg in t.tags for tg in tag)]
            console.print(f"Filtered to [yellow]{len(config.tests)}[/] tests with tags: {tag}")
            
        suite_result = runner.run_suite()
        
        # Display Results Table
        table = Table(title=f"Results: {suite_result.suite_name}")
        table.add_column("Test Case", style="cyan")
        table.add_column("Result", justify="center")
        table.add_column("Cost (USD)", justify="right")
        table.add_column("Duration (ms)", justify="right")
        table.add_column("Diffs/Details")
        
        for res in suite_result.results:
            result_str = "[green]PASSED[/]" if res.result == TestResult.PASSED else \
                         "[red]FAILED[/]" if res.result == TestResult.FAILED else \
                         "[bold red]ERROR[/]"
            
            details = []
            if res.error_message:
                details.append(f"[red]{res.error_message}[/]")
            
            if res.assertion_results:
                for r in res.assertion_results:
                    if not r['passed']:
                        details.append(r['message'])
            
            # Add Diff Details
            if res.diffs:
                for d in res.diffs:
                    color = "red" if d.severity == "error" else "yellow"
                    details.append(f"[{color}]{d.message}[/]")
                
            table.add_row(
                res.test_name,
                result_str,
                f"${res.trace.total_cost_usd:.4f}",
                f"{res.duration_ms:.1f}",
                "\n".join(details)
            )
            
        console.print(table)
        
        # Summary
        console.print(f"\n[bold]Summary:[/] [green]{suite_result.total_passed} Passed[/], "
                      f"[red]{suite_result.total_failed} Failed[/], "
                      f"[bold red]{suite_result.total_errors} Errors[/]")
        console.print(f"Total Cost: [bold]${suite_result.total_cost_usd:.4f}[/]")
        console.print(f"Total Duration: [bold]{suite_result.duration_ms:.1f}ms[/]")
        
        # Check fail-on-cost
        if fail_on_cost and suite_result.total_cost_usd > fail_on_cost:
             console.print(f"[bold red]FAILURE:[/] Total cost ${suite_result.total_cost_usd:.4f} "
                           f"exceeds limit ${fail_on_cost:.4f}")
             if ci:
                 import sys
                 sys.exit(1)
        
        if ci and (suite_result.total_failed > 0 or suite_result.total_errors > 0):
            import sys
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if ci:
            import sys
            sys.exit(1)


@cli.command()
@click.argument('test_name')
@click.option('--suite', '-s', default='agentci.yaml')
@click.option('--output', '-o', help='Output path for golden trace')
def record(test_name, suite, output):
    """Run agent live and save the trace as a golden baseline."""
    # Execute the test → save trace as golden JSON
    pass


@cli.command()
@click.argument('test_name')
@click.option('--suite', '-s', default='agentci.yaml')
def diff(test_name, suite):
    """Show diff between latest run and golden trace."""
    pass


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', type=click.Path(), required=True)
def report(input, output):
    """Generate an HTML report from a JSON results file."""
    pass


if __name__ == '__main__':
    cli()
