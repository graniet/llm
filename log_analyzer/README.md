# Log Analyzer

This Rust program analyzes log files for errors, warnings, panics, critical errors, and unfinished work indicators.

## Features

- Detects various levels of issues: Critical Errors, Panics, Errors, Warnings, and Unfinished Work (TODO/FIXME).
- Provides a summary of findings.
- Lists detailed occurrences of each finding with line numbers.

## Building and Running with Nix

To build and run this project using Nix flakes, ensure you have Nix installed and flakes enabled.

1.  **Build the project:**

    ```bash
    nix build .#log-analyzer
    ```

    This will build the `log-analyzer` executable. You can find it in `result/bin/log-analyzer` after a successful build.

2.  **Run the analyzer:**

    You can run the analyzer directly using `nix run`:

    ```bash
    nix run .#log-analyzer -- --log-file <path/to/your/log/file.log>
    ```

    For example, to analyze a `telemetry.log` file:

    ```bash
    nix run .#log-analyzer -- --log-file ~/today/llm/logs/telemetry.log
    ```

3.  **Enter a development shell:**

    To enter a development environment with the Rust toolchain and other useful tools:

    ```bash
    nix develop
    ```

    Inside the development shell, you can use `cargo build`, `cargo run`, `rustc`, `rustfmt`, `clippy`, etc.

## Usage

```bash
log_analyzer --log-file <path/to/your/log/file.log>
```

### Arguments

-   `--log-file <path>`: Path to the log file to analyze. (Required)

## Example Output

```
Analyzing log file: /path/to/your/log/file.log
------------------------------------
[CRITICAL] Line 10: CRITICAL: System failure detected.
[ERROR] Line 25: Error: Failed to connect to database.
[WARNING] Line 40: Warning: Disk space low.
[UNFINISHED WORK] Line 50: TODO: Implement retry logic.
------------------------------------
Analysis Summary:
  Critical Errors found: 1
  Panics found: 0
  Errors found: 1
  Warnings found: 1
  Unfinished work found: 1

Detailed Findings:
[CRITICAL] Line 10: CRITICAL: System failure detected.
[ERROR] Line 25: Error: Failed to connect to database.
[WARNING] Line 40: Warning: Disk space low.
[UNFINISHED WORK] Line 50: TODO: Implement retry logic.
```
