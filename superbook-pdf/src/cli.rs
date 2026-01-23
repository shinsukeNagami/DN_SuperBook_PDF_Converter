//! CLI interface module
//!
//! Provides command-line interface using clap derive macros.

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

/// High-quality PDF converter for scanned books
#[derive(Parser, Debug)]
#[command(name = "superbook-pdf")]
#[command(author = "DN_SuperBook_PDF_Converter Contributors")]
#[command(version)]
#[command(about = "High-quality PDF converter for scanned books", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Convert PDF files with AI enhancement
    Convert(ConvertArgs),
    /// Show system information
    Info,
}

/// Arguments for the convert command
#[derive(clap::Args, Debug)]
pub struct ConvertArgs {
    /// Input PDF file or directory
    pub input: PathBuf,

    /// Output directory
    #[arg(default_value = "./output")]
    pub output: PathBuf,

    /// Enable Japanese OCR (YomiToku)
    #[arg(short, long)]
    pub ocr: bool,

    /// Enable AI upscaling (RealESRGAN)
    #[arg(short, long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    pub upscale: bool,

    /// Disable AI upscaling
    #[arg(long = "no-upscale")]
    #[arg(action = clap::ArgAction::SetTrue)]
    no_upscale: bool,

    /// Enable deskew correction
    #[arg(short, long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    pub deskew: bool,

    /// Disable deskew correction
    #[arg(long = "no-deskew")]
    #[arg(action = clap::ArgAction::SetTrue)]
    no_deskew: bool,

    /// Margin trim percentage
    #[arg(short, long, default_value_t = 0.5)]
    pub margin_trim: f32,

    /// Output DPI
    #[arg(long, default_value_t = 300)]
    pub dpi: u32,

    /// Number of parallel threads
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Enable GPU processing
    #[arg(short, long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    pub gpu: bool,

    /// Disable GPU processing
    #[arg(long = "no-gpu")]
    #[arg(action = clap::ArgAction::SetTrue)]
    no_gpu: bool,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Suppress progress output
    #[arg(short, long)]
    pub quiet: bool,

    /// Show execution plan without processing
    #[arg(long)]
    pub dry_run: bool,
}

impl ConvertArgs {
    /// Get effective upscale setting (considering --no-upscale flag)
    pub fn effective_upscale(&self) -> bool {
        self.upscale && !self.no_upscale
    }

    /// Get effective deskew setting (considering --no-deskew flag)
    pub fn effective_deskew(&self) -> bool {
        self.deskew && !self.no_deskew
    }

    /// Get effective GPU setting (considering --no-gpu flag)
    pub fn effective_gpu(&self) -> bool {
        self.gpu && !self.no_gpu
    }

    /// Get thread count (default to available CPUs)
    pub fn thread_count(&self) -> usize {
        self.threads.unwrap_or_else(num_cpus::get)
    }
}

/// Create a styled progress bar for file processing
pub fn create_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("Invalid progress bar template")
            .progress_chars("#>-"),
    );
    pb
}

/// Create a spinner for indeterminate progress
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Invalid spinner template"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Create a progress bar for page processing
pub fn create_page_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] Page {pos}/{len} ({percent}%) - {msg}")
            .expect("Invalid progress bar template")
            .progress_chars("█▓░"),
    );
    pb
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parse() {
        // Verify CLI can be built
        Cli::command().debug_assert();
    }

    #[test]
    fn test_help_display() {
        let mut cmd = Cli::command();
        let help = cmd.render_help().to_string();
        assert!(help.contains("superbook-pdf"));
        assert!(help.contains("convert"));
    }

    #[test]
    fn test_version_display() {
        let cmd = Cli::command();
        let version = cmd.get_version().unwrap_or("unknown");
        assert!(!version.is_empty());
    }

    #[test]
    fn test_missing_input_error() {
        let result = Cli::try_parse_from(["superbook-pdf", "convert"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("required"));
    }

    #[test]
    fn test_option_parsing() {
        let cli = Cli::try_parse_from([
            "superbook-pdf",
            "convert",
            "input.pdf",
            "--ocr",
            "--no-upscale",
            "--dpi",
            "600",
            "-vvv",
        ])
        .unwrap();

        if let Commands::Convert(args) = cli.command {
            assert!(args.ocr);
            assert!(!args.effective_upscale());
            assert_eq!(args.dpi, 600);
            assert_eq!(args.verbose, 3);
        } else {
            panic!("Expected Convert command");
        }
    }

    #[test]
    fn test_default_values() {
        let cli = Cli::try_parse_from(["superbook-pdf", "convert", "input.pdf"]).unwrap();

        if let Commands::Convert(args) = cli.command {
            assert!(!args.ocr);
            assert!(args.effective_upscale());
            assert!(args.effective_deskew());
            assert_eq!(args.margin_trim, 0.5);
            assert_eq!(args.dpi, 300);
            assert!(args.effective_gpu());
            assert_eq!(args.verbose, 0);
            assert!(!args.quiet);
            assert!(!args.dry_run);
        } else {
            panic!("Expected Convert command");
        }
    }

    #[test]
    fn test_directory_input() {
        let cli = Cli::try_parse_from(["superbook-pdf", "convert", "/tmp/test_dir", "--dry-run"])
            .unwrap();

        if let Commands::Convert(args) = cli.command {
            assert!(args.dry_run);
        } else {
            panic!("Expected Convert command");
        }
    }

    #[test]
    fn test_info_command() {
        let cli = Cli::try_parse_from(["superbook-pdf", "info"]).unwrap();

        assert!(matches!(cli.command, Commands::Info));
    }

    // TC-CLI-007: Progress bar display
    #[test]
    fn test_progress_bar_display() {
        // Test that progress bar can be created and styled
        let pb = create_progress_bar(100);
        assert_eq!(pb.length(), Some(100));

        // Test progress updates
        pb.set_position(50);
        assert_eq!(pb.position(), 50);

        pb.finish_with_message("done");
    }

    #[test]
    fn test_spinner_creation() {
        let spinner = create_spinner("Processing...");
        assert_eq!(spinner.message(), "Processing...");
        spinner.finish_with_message("Complete");
    }

    #[test]
    fn test_page_progress_bar() {
        let pb = create_page_progress_bar(10);
        assert_eq!(pb.length(), Some(10));

        for i in 0..10 {
            pb.set_position(i);
            pb.set_message(format!("page_{}.png", i));
        }
        pb.finish_with_message("All pages processed");
    }
}
