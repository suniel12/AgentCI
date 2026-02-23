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

import os
import sys
import shutil
import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

from .config import load_config
from .runner import TestRunner
from .models import TestResult

console = Console()

@click.group()
@click.version_option()
def cli():
    """Agent CI — Continuous Integration for AI Agents"""
    # Load .env variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Ensure current directory is in sys.path so we can import agents
    import sys
    import os
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    pass


@cli.command()
def init():
    """Scaffold a new Agent CI test suite."""

    target_config = "agentci.yaml"
    target_golden_dir = "golden"
    target_golden_file = os.path.join(target_golden_dir, "demo_test.golden.json")

    AGENTCI_YAML_TEMPLATE = """name: "Agent CI Demo Suite"
description: "A simple suite to verify Agent CI works"

tests:
  - name: "demo_test"
    description: "Verify the mock flight search agent returns results"
    golden_trace: "golden/demo_test.golden.json"
    tags: ["demo"]
    assertions:
      - type: "tool_called"
        tool: "search_flights"
      - type: "cost_under"
        threshold: 0.05
"""
    
    GOLDEN_JSON_TEMPLATE = """{
  "trace_id": "trace_demo_123",
  "total_cost_usd": 0.001,
  "total_tokens": 150,
  "total_duration_ms": 45.0,
  "total_llm_calls": 1,
  "total_tool_calls": 1,
  "spans": [
    {
      "span_id": "span_1",
      "name": "mock_agent_run",
      "tool_calls": [
        {
          "tool_name": "search_flights",
          "arguments": {"origin": "SFO", "destination": "JFK"}
        }
      ]
    }
  ]
}"""

    # 1. Create agentci.yaml
    if os.path.exists(target_config):
        console.print(f"[yellow]Warning:[/] {target_config} already exists.")
        if not Confirm.ask("Overwrite it?", default=False):
            console.print("Skipped config creation.")
        else:
            with open(target_config, "w") as f:
                f.write(AGENTCI_YAML_TEMPLATE)
            console.print(f"[green]Created {target_config}[/]")
    else:
        with open(target_config, "w") as f:
            f.write(AGENTCI_YAML_TEMPLATE)
        console.print(f"[green]Created {target_config}[/]")

    # 2. Create golden directory and sample trace
    if not os.path.exists(target_golden_dir):
        os.makedirs(target_golden_dir)
        console.print(f"[green]Created {target_golden_dir}/[/]")
    
    if os.path.exists(target_golden_file):
        pass
    else:
        with open(target_golden_file, "w") as f:
            f.write(GOLDEN_JSON_TEMPLATE)
        console.print(f"[green]Created {target_golden_file}[/]")

    console.print("\\n[bold green]Init complete![/]")
    console.print("Next steps:")
    console.print("1. Edit [cyan]agentci.yaml[/] to point to your agent function.")
    console.print("2. Run [cyan]agentci run[/] to see the demo test pass (or fail if agent not found).")


from collections import defaultdict
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
            
        suite_result = runner.run_suite(runs=runs)
        
        # Display Results Table
        table = Table(title=f"Results: {suite_result.suite_name}")
        
        if runs > 1:
            # Statistical Display
            table.add_column("Test Case", style="cyan")
            table.add_column("Pass Rate", justify="center")
            table.add_column("Mean Cost", justify="right")
            table.add_column("Mean Duration", justify="right")
            table.add_column("Status")

            # Group by test name
            from collections import defaultdict
            grouped_results = defaultdict(list)
            for res in suite_result.results:
                grouped_results[res.test_name].append(res)
            
            for test_name, results in grouped_results.items():
                passed_count = sum(1 for r in results if r.result == TestResult.PASSED)
                pass_rate = (passed_count / len(results)) * 100
                mean_cost = sum(r.trace.total_cost_usd for r in results) / len(results)
                mean_duration = sum(r.duration_ms for r in results) / len(results)
                
                status_style = "green" if passed_count == len(results) else "yellow" if passed_count > 0 else "red"
                status_str = "STABLE" if passed_count == len(results) else "FLAKY" if passed_count > 0 else "FAILING"
                
                table.add_row(
                    test_name,
                    f"{passed_count}/{len(results)} ({pass_rate:.0f}%)",
                    f"${mean_cost:.4f}",
                    f"{mean_duration:.1f}ms",
                    f"[{status_style}]{status_str}[/]"
                )
        else:
            # Single Run Display (Existing logic)
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
    try:
        config = load_config(suite)
        runner = TestRunner(config)
        agent_fn = runner._import_agent()
        
        # Find the specific test
        test = next((t for t in config.tests if t.name == test_name), None)
        if not test:
            console.print(f"[bold red]Error:[/] Test '{test_name}' not found in {suite}")
            return
            
        console.print(f"Recording trace for [cyan]{test_name}[/]...")
        
        # Run the test
        result = runner.run_test(test, agent_fn)
        
        # Show summary
        console.print(f"Duration: {result.duration_ms:.1f}ms")
        console.print(f"Cost: ${result.trace.total_cost_usd:.4f}")
        console.print(f"Tool Calls: {len(result.trace.tool_call_sequence)}")
        
        if result.error_message:
             console.print(f"[bold red]Error during run:[/] {result.error_message}")
             # We might still want to save it if it's a valid trace of a failure?
             # detailed prompt below
        
        # Determine output path
        import os
        if output:
            save_path = output
        elif test.golden_trace:
            save_path = test.golden_trace
        else:
            save_path = f"golden/{test_name}.golden.json"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Prompt user
        from rich.prompt import Confirm
        if Confirm.ask(f"Save golden trace to [yellow]{save_path}[/]?", default=True):
            with open(save_path, 'w') as f:
                f.write(result.trace.model_dump_json(indent=2))
            console.print(f"[green]Saved![/]")
        else:
            console.print("[yellow]Cancelled.[/]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")


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
