use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use std::io::Cursor;
use std::sync::Arc;
const MAX_RECORD_SECONDS: u32 = 60;
const I16_SCALE: f32 = 1.0 / i16::MAX as f32;
const U16_SCALE: f32 = 1.0 / u16::MAX as f32;
const U16_RANGE: f32 = 2.0;
const U16_OFFSET: f32 = 1.0;
const WAV_BITS: u16 = 32;
const WAV_CHANNELS: u16 = 1;
#[derive(Clone)]
struct SampleBuffer {
    samples: Arc<Mutex<Vec<f32>>>,
    channels: usize,
    max_samples: usize,
}
impl SampleBuffer {
    fn new(samples: Arc<Mutex<Vec<f32>>>, channels: usize, max_samples: usize) -> Self {
        Self {
            samples,
            channels,
            max_samples,
        }
    }
    fn push_f32(&self, input: &[f32]) {
        let iter = input.iter().step_by(self.channels).copied();
        self.push_samples(iter);
    }
    fn push_i16(&self, input: &[i16]) {
        let iter = input
            .iter()
            .step_by(self.channels)
            .map(|s| *s as f32 * I16_SCALE);
        self.push_samples(iter);
    }
    fn push_u16(&self, input: &[u16]) {
        let iter = input
            .iter()
            .step_by(self.channels)
            .map(|s| (*s as f32 * U16_SCALE) * U16_RANGE - U16_OFFSET);
        self.push_samples(iter);
    }
    fn push_samples<I>(&self, input: I)
    where
        I: Iterator<Item = f32>,
    {
        let mut guard = self.samples.lock();
        let remaining = self.max_samples.saturating_sub(guard.len());
        if remaining == 0 {
            return;
        }
        guard.extend(input.take(remaining));
    }
}
pub struct AudioRecorder {
    device: cpal::Device,
    config: cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    samples: Arc<Mutex<Vec<f32>>>,
    stream: Option<cpal::Stream>,
    channels: usize,
    max_samples: usize,
}
impl AudioRecorder {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no input device available")?;
        let supported = device
            .default_input_config()
            .context("no default input config")?;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let channels = usize::from(config.channels);
        if channels == 0 {
            bail!("input device reported zero channels");
        }
        let max_samples = compute_max_samples(config.sample_rate.0)?;
        let samples = Arc::new(Mutex::new(Vec::with_capacity(max_samples)));
        Ok(Self {
            device,
            config,
            sample_format,
            samples,
            stream: None,
            channels,
            max_samples,
        })
    }
    pub fn start(&mut self) -> Result<()> {
        if self.stream.is_some() {
            bail!("recording already started");
        }
        let stream = self.build_stream()?;
        stream.play().context("failed to start audio stream")?;
        self.stream = Some(stream);
        Ok(())
    }
    pub fn stop(&mut self) -> Vec<f32> {
        self.stream.take();
        self.take_samples()
    }
    pub fn clear(&self) {
        self.samples.lock().clear();
    }
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate.0
    }
    fn take_samples(&self) -> Vec<f32> {
        let mut guard = self.samples.lock();
        std::mem::take(&mut *guard)
    }
    fn build_stream(&self) -> Result<cpal::Stream> {
        let buffer = SampleBuffer::new(Arc::clone(&self.samples), self.channels, self.max_samples);
        match self.sample_format {
            cpal::SampleFormat::F32 => self.build_f32_stream(buffer),
            cpal::SampleFormat::I16 => self.build_i16_stream(buffer),
            cpal::SampleFormat::U16 => self.build_u16_stream(buffer),
            _ => bail!("unsupported input sample format"),
        }
    }
    fn build_f32_stream(&self, buffer: SampleBuffer) -> Result<cpal::Stream> {
        let buffer = buffer.clone();
        self.device
            .build_input_stream(
                &self.config,
                move |data: &[f32], _| buffer.push_f32(data),
                log_stream_error,
                None,
            )
            .context("failed to build f32 input stream")
    }
    fn build_i16_stream(&self, buffer: SampleBuffer) -> Result<cpal::Stream> {
        let buffer = buffer.clone();
        self.device
            .build_input_stream(
                &self.config,
                move |data: &[i16], _| buffer.push_i16(data),
                log_stream_error,
                None,
            )
            .context("failed to build i16 input stream")
    }
    fn build_u16_stream(&self, buffer: SampleBuffer) -> Result<cpal::Stream> {
        let buffer = buffer.clone();
        self.device
            .build_input_stream(
                &self.config,
                move |data: &[u16], _| buffer.push_u16(data),
                log_stream_error,
                None,
            )
            .context("failed to build u16 input stream")
    }
}
pub async fn pcm_to_wav_bytes(samples: Vec<f32>, sample_rate: u32) -> Result<Vec<u8>> {
    tokio::task::spawn_blocking(move || encode_wav(samples, sample_rate))
        .await
        .context("wav encoding task failed")?
}
fn compute_max_samples(sample_rate: u32) -> Result<usize> {
    let rate = usize::try_from(sample_rate).context("sample rate overflow")?;
    let secs = usize::try_from(MAX_RECORD_SECONDS).context("max seconds overflow")?;
    rate.checked_mul(secs).context("max sample count overflow")
}
fn encode_wav(samples: Vec<f32>, sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: WAV_CHANNELS,
        sample_rate,
        bits_per_sample: WAV_BITS,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).context("wav writer init failed")?;
    for sample in samples {
        writer
            .write_sample(sample)
            .context("wav writer sample failed")?;
    }
    writer.finalize().context("wav writer finalize failed")?;
    Ok(cursor.into_inner())
}
fn log_stream_error(err: cpal::StreamError) {
    eprintln!("Audio input error: {err}");
}
