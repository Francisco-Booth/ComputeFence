import sys
import json
from rich.console import Console
from rich.markup import escape
from computefence import __version__
from computefence.checks.environment import check_environment
from computefence.checks.storage import check_disk_headroom, check_output_dir, check_storage
from computefence.checks.dataset import check_dataset
from computefence.telemetry import record_run
from computefence.drift import save_env, check_drift

console = Console()

def print_result(result):
    status = result.get("status")
    message = escape(result.get("message", ""))
    fix = result.get("fix")
    if status == "pass":
        console.print(f"  [green]✓[/green] {message}")
    elif status == "warn":
        console.print(f"  [yellow]⚠[/yellow] {message}")
        if fix:
            console.print(f"    [dim]Fix: {escape(fix)}[/dim]")
    elif status == "fail":
        console.print(f"  [red]✗[/red] {message}")
        if fix:
            console.print(f"    [dim]Fix: {escape(fix)}[/dim]")

def doctor():
    dataset = None
    input_column = None
    label_column = None
    output_dir = None
    save_env_flag = False
    check_drift_flag = False
    path_override = None
    json_flag = False
    skip_ids = set()

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
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif args[i] == "--save-env":
            save_env_flag = True
            i += 1
        elif args[i] == "--check-drift":
            check_drift_flag = True
            i += 1
        elif args[i] == "--path" and i + 1 < len(args):
            path_override = args[i + 1]
            i += 2
        elif args[i] == "--json":
            json_flag = True
            i += 1
        elif args[i] == "--skip" and i + 1 < len(args):
            skip_ids = set(args[i + 1].split(","))
            i += 2
        else:
            i += 1

    if not json_flag:
        console.print()
        console.print(f"[bold]ComputeFence v{__version__} — Pre-flight diagnostic[/bold]")
        console.print("━" * 50)

    env_results = check_environment()
    storage_results = check_storage() + check_disk_headroom() + check_output_dir(output_dir)
    dataset_results = check_dataset(
        dataset_path=dataset,
        input_column=input_column,
        label_column=label_column,
    )

    all_results = env_results + storage_results + dataset_results

    if check_drift_flag:
        drift_results = check_drift(output_dir=output_dir, path_override=path_override)
        all_results = all_results + drift_results
    else:
        drift_results = []

    # Apply --skip filter
    if skip_ids:
        all_results = [r for r in all_results if r.get("id") not in skip_ids]
        env_results = [r for r in env_results if r.get("id") not in skip_ids]
        storage_results = [r for r in storage_results if r.get("id") not in skip_ids]
        dataset_results = [r for r in dataset_results if r.get("id") not in skip_ids]
        drift_results = [r for r in drift_results if r.get("id") not in skip_ids]

    failures = [r for r in all_results if r["status"] == "fail"]
    warnings = [r for r in all_results if r["status"] == "warn"]
    passed = [r for r in all_results if r["status"] == "pass"]

    if json_flag:
        output = {
            "schema_version": "1",
            "computefence_version": __version__,
            "summary": {
                "passed": len(passed),
                "warnings": len(warnings),
                "blockers": len(failures),
            },
            "checks": [
                {
                    "id": r["id"],
                    "status": r["status"],
                    "message": r.get("message", ""),
                    "remediation": r.get("fix", ""),
                }
                for r in all_results
            ],
        }
        print(json.dumps(output, indent=2))
        record_run(all_results)
        if failures:
            sys.exit(2)
        elif warnings:
            sys.exit(1)
        sys.exit(0)

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

    if check_drift_flag:
        console.print()
        console.print("[bold blue]Drift[/bold blue]")
        for result in drift_results:
            print_result(result)

    console.print()
    console.print("━" * 50)
    if failures:
        console.print("[red]" + str(len(failures)) + " error(s) and " + str(len(warnings)) + " warning(s) found. Fix errors before launching.[/red]")
    elif warnings:
        console.print("[yellow]" + str(len(warnings)) + " warning(s) found. Review before launching.[/yellow]")
    else:
        console.print("[green]All checks passed. Safe to launch.[/green]")
        console.print()
        console.print("[dim]Tip: add this to your launch script to run automatically every time:[/dim]")
        console.print('[dim]  alias train="computefence doctor && python train.py"[/dim]')
    console.print(
        "[dim]Pseudonymous run stats are not collected by default. "
        "To opt in: touch ~/.computefence_telemetry[/dim]"
    )
    console.print()

    if save_env_flag:
        if failures:
            console.print("[yellow]⚠ --save-env skipped: blockers present. Fix errors first.[/yellow]")
            console.print()
        else:
            success, msg = save_env(output_dir=output_dir, path_override=path_override)
            if success:
                console.print(f"[green]✓ Baseline saved to {escape(msg)}[/green]")
                console.print("[dim]Commit this file to your training repo. On any new pod, clone your repo and run --check-drift to verify the environment matches your last known-good run.[/dim]")
            else:
                console.print(f"[yellow]⚠ Could not save fingerprint: {escape(msg)}[/yellow]")
            console.print()

    record_run(all_results)

    if failures:
        sys.exit(2)
    elif warnings:
        sys.exit(1)

def main():
    args = sys.argv[1:]
    if not args or args[0] == "doctor":
        doctor()
    else:
        console.print(f"[red]Unknown command: {escape(args[0])}[/red]")
        console.print("Usage: computefence doctor")
        sys.exit(64)

if __name__ == "__main__":
    main()