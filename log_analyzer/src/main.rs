use std::fs::File;
use std::io::{self, BufReader, BufRead};
use clap::Parser;
use regex::Regex;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the log file to analyze
    #[arg(short, long)]
    log_file: String,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let file_path = &args.log_file;
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    println!("Analyzing log file: {}", file_path);
    println!("------------------------------------");

    let error_re = Regex::new(r"(?i)error|fail|exception|denied").unwrap();
    let warning_re = Regex::new(r"(?i)warn|warning").unwrap();
    let critical_re = Regex::new(r"(?i)critical|fatal").unwrap();
    let unfinished_re = Regex::new(r"(?i)unfinished work|todo|fixme").unwrap();
    let panic_re = Regex::new(r"(?i)panic").unwrap();

    let mut findings: Vec<LogFinding> = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let current_line_num = line_num + 1;

        if critical_re.is_match(&line) {
            findings.push(LogFinding {
                line_num: current_line_num,
                line_content: line.clone(),
                finding_type: FindingType::CriticalError,
            });
        } else if panic_re.is_match(&line) {
            findings.push(LogFinding {
                line_num: current_line_num,
                line_content: line.clone(),
                finding_type: FindingType::Panic,
            });
        } else if error_re.is_match(&line) {
            findings.push(LogFinding {
                line_num: current_line_num,
                line_content: line.clone(),
                finding_type: FindingType::Error,
            });
        } else if warning_re.is_match(&line) {
            findings.push(LogFinding {
                line_num: current_line_num,
                line_content: line.clone(),
                finding_type: FindingType::Warning,
            });
        } else if unfinished_re.is_match(&line) {
            findings.push(LogFinding {
                line_num: current_line_num,
                line_content: line.clone(),
                finding_type: FindingType::UnfinishedWork,
            });
        }
    }

    print_summary(&findings);

    Ok(())
}

#[derive(Debug)]
enum FindingType {
    CriticalError,
    Panic,
    Error,
    Warning,
    UnfinishedWork,
}

#[derive(Debug)]
struct LogFinding {
    line_num: usize,
    line_content: String,
    finding_type: FindingType,
}

fn print_summary(findings: &[LogFinding]) {
    println!("------------------------------------");
    println!("Analysis Summary:");

    let mut critical_count = 0;
    let mut panic_count = 0;
    let mut error_count = 0;
    let mut warning_count = 0;
    let mut unfinished_count = 0;

    for finding in findings {
        match finding.finding_type {
            FindingType::CriticalError => critical_count += 1,
            FindingType::Panic => panic_count += 1,
            FindingType::Error => error_count += 1,
            FindingType::Warning => warning_count += 1,
            FindingType::UnfinishedWork => unfinished_count += 1,
        }
    }

    println!("  Critical Errors found: {}", critical_count);
    println!("  Panics found: {}", panic_count);
    println!("  Errors found: {}", error_count);
    println!("  Warnings found: {}", warning_count);
    println!("  Unfinished work found: {}", unfinished_count);
    println!("");

    if !findings.is_empty() {
        println!("Detailed Findings:");
        for finding in findings {
            match finding.finding_type {
                FindingType::CriticalError => println!("[CRITICAL] Line {}: {}", finding.line_num, finding.line_content),
                FindingType::Panic => println!("[PANIC] Line {}: {}", finding.line_num, finding.line_content),
                FindingType::Error => println!("[ERROR] Line {}: {}", finding.line_num, finding.line_content),
                FindingType::Warning => println!("[WARNING] Line {}: {}", finding.line_num, finding.line_content),
                FindingType::UnfinishedWork => println!("[UNFINISHED WORK] Line {}: {}", finding.line_num, finding.line_content),
            }
        }
    } else {
        println!("No significant findings.");
    }
}