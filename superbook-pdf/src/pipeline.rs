//! Pipeline processing module
//!
//! Provides a clean API for PDF processing pipeline, separating
//! business logic from CLI handling.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;
use thiserror::Error;

use crate::cli::ConvertArgs;

/// Pipeline processing error
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Input file not found: {0}")]
    InputNotFound(PathBuf),

    #[error("Output directory not writable: {0}")]
    OutputNotWritable(PathBuf),

    #[error("PDF extraction failed: {0}")]
    ExtractionFailed(String),

    #[error("Image processing failed: {0}")]
    ImageProcessingFailed(String),

    #[error("PDF generation failed: {0}")]
    PdfGenerationFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Output DPI
    pub dpi: u32,
    /// Enable deskew
    pub deskew: bool,
    /// Margin trim percentage
    pub margin_trim: f64,
    /// Enable AI upscaling
    pub upscale: bool,
    /// Enable GPU
    pub gpu: bool,
    /// Enable internal resolution normalization
    pub internal_resolution: bool,
    /// Enable color correction
    pub color_correction: bool,
    /// Enable offset alignment
    pub offset_alignment: bool,
    /// Output height
    pub output_height: u32,
    /// Enable OCR
    pub ocr: bool,
    /// Max pages for debug
    pub max_pages: Option<usize>,
    /// Save debug images
    pub save_debug: bool,
    /// JPEG quality (0-100)
    pub jpeg_quality: u8,
    /// Thread count (None = auto)
    pub threads: Option<usize>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            dpi: 300,
            deskew: true,
            margin_trim: 0.5,
            upscale: true,
            gpu: true,
            internal_resolution: false,
            color_correction: false,
            offset_alignment: false,
            output_height: 3508,
            ocr: false,
            max_pages: None,
            save_debug: false,
            jpeg_quality: 90,
            threads: None,
        }
    }
}

impl PipelineConfig {
    /// Create configuration from CLI convert arguments
    pub fn from_convert_args(args: &ConvertArgs) -> Self {
        let advanced = args.advanced;
        Self {
            dpi: args.dpi,
            deskew: args.effective_deskew(),
            margin_trim: args.margin_trim as f64,
            upscale: args.effective_upscale(),
            gpu: args.effective_gpu(),
            internal_resolution: args.internal_resolution || advanced,
            color_correction: args.color_correction || advanced,
            offset_alignment: args.offset_alignment || advanced,
            output_height: args.output_height,
            ocr: args.ocr,
            max_pages: args.max_pages,
            save_debug: args.save_debug,
            jpeg_quality: args.jpeg_quality,
            threads: args.threads,
        }
    }

    /// Convert to JSON string for cache digest
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Builder pattern: set DPI
    pub fn with_dpi(mut self, dpi: u32) -> Self {
        self.dpi = dpi;
        self
    }

    /// Builder pattern: set deskew
    pub fn with_deskew(mut self, enabled: bool) -> Self {
        self.deskew = enabled;
        self
    }

    /// Builder pattern: set margin trim
    pub fn with_margin_trim(mut self, percent: f64) -> Self {
        self.margin_trim = percent;
        self
    }

    /// Builder pattern: set upscale
    pub fn with_upscale(mut self, enabled: bool) -> Self {
        self.upscale = enabled;
        self
    }

    /// Builder pattern: set GPU
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.gpu = enabled;
        self
    }

    /// Builder pattern: set OCR
    pub fn with_ocr(mut self, enabled: bool) -> Self {
        self.ocr = enabled;
        self
    }

    /// Builder pattern: set max pages
    pub fn with_max_pages(mut self, max: Option<usize>) -> Self {
        self.max_pages = max;
        self
    }

    /// Enable all advanced features
    pub fn with_advanced(mut self) -> Self {
        self.internal_resolution = true;
        self.color_correction = true;
        self.offset_alignment = true;
        self
    }
}

/// Result of pipeline processing
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Number of pages processed
    pub page_count: usize,
    /// Detected page number shift
    pub page_number_shift: Option<i32>,
    /// Whether vertical text was detected
    pub is_vertical: bool,
    /// Processing time in seconds
    pub elapsed_seconds: f64,
    /// Output file path
    pub output_path: PathBuf,
    /// Output file size in bytes
    pub output_size: u64,
}

impl PipelineResult {
    /// Create a new pipeline result
    pub fn new(
        page_count: usize,
        page_number_shift: Option<i32>,
        is_vertical: bool,
        elapsed_seconds: f64,
        output_path: PathBuf,
        output_size: u64,
    ) -> Self {
        Self {
            page_count,
            page_number_shift,
            is_vertical,
            elapsed_seconds,
            output_path,
            output_size,
        }
    }

    /// Convert to cache ProcessingResult
    pub fn to_cache_result(&self) -> crate::cache::ProcessingResult {
        crate::cache::ProcessingResult::new(
            self.page_count,
            self.page_number_shift,
            self.is_vertical,
            self.elapsed_seconds,
            self.output_size,
        )
    }
}

/// PDF processing pipeline
pub struct PdfPipeline {
    config: PipelineConfig,
}

impl PdfPipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Get the pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get the output PDF path for a given input PDF
    pub fn get_output_path(&self, input: &Path, output_dir: &Path) -> PathBuf {
        let pdf_name = input.file_stem().unwrap_or_default().to_string_lossy();
        output_dir.join(format!("{}_converted.pdf", pdf_name))
    }

    /// Process a single PDF file
    ///
    /// This is the main entry point for PDF processing.
    /// Currently delegates to the existing main.rs implementation.
    /// Future: Full pipeline implementation.
    pub fn process(&self, input: &Path, output_dir: &Path) -> Result<PipelineResult, PipelineError> {
        let start_time = Instant::now();

        // Validate input
        if !input.exists() {
            return Err(PipelineError::InputNotFound(input.to_path_buf()));
        }

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        let output_path = self.get_output_path(input, output_dir);

        // Read PDF to get page count
        let reader = crate::LopdfReader::new(input)
            .map_err(|e| PipelineError::ExtractionFailed(e.to_string()))?;
        let page_count = reader.info.page_count;

        // For now, return a placeholder result
        // Full processing will be implemented incrementally
        let elapsed = start_time.elapsed().as_secs_f64();

        Ok(PipelineResult::new(
            page_count,
            None,  // page_number_shift
            false, // is_vertical
            elapsed,
            output_path,
            0, // output_size (not generated yet)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ PipelineConfig Tests ============

    #[test]
    fn test_pipeline_config_default() {
        // TC: PIPE-003
        let config = PipelineConfig::default();

        assert_eq!(config.dpi, 300);
        assert!(config.deskew);
        assert_eq!(config.margin_trim, 0.5);
        assert!(config.upscale);
        assert!(config.gpu);
        assert!(!config.internal_resolution);
        assert!(!config.color_correction);
        assert!(!config.offset_alignment);
        assert_eq!(config.output_height, 3508);
        assert!(!config.ocr);
        assert!(config.max_pages.is_none());
        assert!(!config.save_debug);
        assert_eq!(config.jpeg_quality, 90);
        assert!(config.threads.is_none());
    }

    #[test]
    fn test_pipeline_config_to_json() {
        // TC: PIPE-002
        let config = PipelineConfig::default();
        let json = config.to_json();

        assert!(json.contains("\"dpi\":300"));
        assert!(json.contains("\"deskew\":true"));
        assert!(json.contains("\"upscale\":true"));
    }

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::default()
            .with_dpi(600)
            .with_deskew(false)
            .with_upscale(false)
            .with_ocr(true);

        assert_eq!(config.dpi, 600);
        assert!(!config.deskew);
        assert!(!config.upscale);
        assert!(config.ocr);
    }

    #[test]
    fn test_pipeline_config_with_advanced() {
        let config = PipelineConfig::default().with_advanced();

        assert!(config.internal_resolution);
        assert!(config.color_correction);
        assert!(config.offset_alignment);
    }

    #[test]
    fn test_pipeline_config_with_max_pages() {
        let config = PipelineConfig::default().with_max_pages(Some(10));

        assert_eq!(config.max_pages, Some(10));
    }

    #[test]
    fn test_pipeline_config_with_margin_trim() {
        let config = PipelineConfig::default().with_margin_trim(1.0);

        assert_eq!(config.margin_trim, 1.0);
    }

    #[test]
    fn test_pipeline_config_with_gpu() {
        let config = PipelineConfig::default().with_gpu(false);

        assert!(!config.gpu);
    }

    // ============ PipelineResult Tests ============

    #[test]
    fn test_pipeline_result_new() {
        // TC: PIPE-004
        let result = PipelineResult::new(
            100,
            Some(2),
            true,
            45.5,
            PathBuf::from("/output/file.pdf"),
            12345678,
        );

        assert_eq!(result.page_count, 100);
        assert_eq!(result.page_number_shift, Some(2));
        assert!(result.is_vertical);
        assert_eq!(result.elapsed_seconds, 45.5);
        assert_eq!(result.output_path, PathBuf::from("/output/file.pdf"));
        assert_eq!(result.output_size, 12345678);
    }

    #[test]
    fn test_pipeline_result_no_shift() {
        let result = PipelineResult::new(
            50,
            None,
            false,
            10.0,
            PathBuf::from("/output/test.pdf"),
            5000000,
        );

        assert_eq!(result.page_count, 50);
        assert!(result.page_number_shift.is_none());
        assert!(!result.is_vertical);
    }

    #[test]
    fn test_pipeline_result_to_cache() {
        let result = PipelineResult::new(100, Some(2), true, 45.5, PathBuf::from("/out.pdf"), 1000);

        let cache_result = result.to_cache_result();

        assert_eq!(cache_result.page_count, 100);
        assert_eq!(cache_result.page_number_shift, Some(2));
        assert!(cache_result.is_vertical);
        assert_eq!(cache_result.elapsed_seconds, 45.5);
        assert_eq!(cache_result.output_size, 1000);
    }

    // ============ PdfPipeline Tests ============

    #[test]
    fn test_pdf_pipeline_new() {
        // TC: PIPE-005
        let config = PipelineConfig::default();
        let pipeline = PdfPipeline::new(config);

        assert_eq!(pipeline.config().dpi, 300);
    }

    #[test]
    fn test_pdf_pipeline_get_output_path() {
        let config = PipelineConfig::default();
        let pipeline = PdfPipeline::new(config);

        let input = Path::new("/input/document.pdf");
        let output_dir = Path::new("/output");

        let output_path = pipeline.get_output_path(input, output_dir);

        assert_eq!(output_path, PathBuf::from("/output/document_converted.pdf"));
    }

    #[test]
    fn test_pdf_pipeline_get_output_path_no_extension() {
        let config = PipelineConfig::default();
        let pipeline = PdfPipeline::new(config);

        let input = Path::new("/input/document");
        let output_dir = Path::new("/output");

        let output_path = pipeline.get_output_path(input, output_dir);

        assert_eq!(output_path, PathBuf::from("/output/document_converted.pdf"));
    }

    #[test]
    fn test_pdf_pipeline_process_input_not_found() {
        let config = PipelineConfig::default();
        let pipeline = PdfPipeline::new(config);

        let result = pipeline.process(Path::new("/nonexistent/file.pdf"), Path::new("/output"));

        assert!(matches!(result, Err(PipelineError::InputNotFound(_))));
    }

    // ============ PipelineError Tests ============

    #[test]
    fn test_pipeline_error_display() {
        let err = PipelineError::InputNotFound(PathBuf::from("/test.pdf"));
        assert!(err.to_string().contains("/test.pdf"));

        let err = PipelineError::ExtractionFailed("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_pipeline_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: PipelineError = io_err.into();

        assert!(matches!(err, PipelineError::Io(_)));
    }

    #[test]
    fn test_pipeline_error_variants() {
        let errors = vec![
            PipelineError::InputNotFound(PathBuf::from("/test.pdf")),
            PipelineError::OutputNotWritable(PathBuf::from("/out")),
            PipelineError::ExtractionFailed("test".to_string()),
            PipelineError::ImageProcessingFailed("test".to_string()),
            PipelineError::PdfGenerationFailed("test".to_string()),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }
}
