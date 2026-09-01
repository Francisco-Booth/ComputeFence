import sys
from rich.console import Console
from computefence import __version__
from computefence.checks.environment import check_environment
from computefence.checks.storage import check_disk_headroom, check_storage
from computefence.checks.dataset import check_dataset
from computefence.telemetry import record_run

console = Console()

def print_result(result):
    status = result.get("status")
    message = result.get("message")
    fix = result.get("fix")
    if status == "pass":
        console.print(f"  [green]✓[/green] {message}")
    elif status == "warn":
        console.print(f"  [yellow]⚠[/yellow] {message}")
        if fix:
            console.print(f"    [dim]Fix: {fix}[/dim]")
    elif status == "fail":
        console.print(f"  [red]✗[/red] {message}")
        if fix:
            console.print(f"    [dim]Fix: {fix}[/dim]")

def doctor():
    dataset = None
    input_column = None
    label_column = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dataset" and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
        elif args[i] == "--input-column" and i + 1 < len(args):
            input_column = args[i + 1]
            i += 2
        elif args[i] == "--label-column" and i + 1 < len(args):
            label_column = args[i + 1]
            i += 2
        else:
            i += 1

    console.print()
    console.print(f"[bold]ComputeFence v{__version__} — Pre-flight diagnostic[/bold]")
    console.print("━" * 50)

    env_results = check_environment()
    storage_results = check_storage() + check_disk_headroom()
    dataset_results = check_dataset(
        dataset_path=dataset,
        input_column=input_column,
        label_column=label_column,
    )

    all_results = env_results + storage_results + dataset_results

    failures = [r for r in all_results if r["status"] == "fail"]
    warnings = [r for r in all_results if r["status"] == "warn"]
    passed = [r for r in all_results if r["status"] == "pass"]

    console.print()
    console.print(
        f"[yellow]{len(warnings)} WARNINGS[/yellow]  ·  "
        f"[red]{len(failures)} BLOCKERS[/red]  ·  "
        f"[green]{len(passed)} PASSED[/green]"
    )
    console.print()
    console.print("[bold blue]Environment[/bold blue]")
    for result in env_results:
        print_result(result)
    console.print()
    console.print("[bold blue]Storage[/bold blue]")
    for result in storage_results:
        print_result(result)
    console.print()
    console.print("[bold blue]Dataset[/bold blue]")
    for result in dataset_results:
        print_result(result)
    console.print()
    console.print("━" * 50)
    if failures:
        console.print("[red]" + str(len(failures)) + " error(s) and " + str(len(warnings)) + " warning(s) found. Fix errors before launching.[/red]")
    elif warnings:
        console.print("[yellow]" + str(len(warnings)) + " warning(s) found. Review before launching.[/yellow]")
    else:
        console.print("[green]All checks passed. Safe to launch.[/green]")
    console.print(
        "[dim]Anonymous run stats are collected to improve ComputeFence. "
        "To opt out: touch ~/.computefence_no_telemetry[/dim]"
    )
    console.print()

    record_run(all_results)

def main():
    args = sys.argv[1:]
    if not args or args[0] == "doctor":
        doctor()
    else:
        console.print(f"[red]Unknown command: {args[0]}[/red]")
        console.print("Usage: computefence doctor")

if __name__ == "__main__":
    main()