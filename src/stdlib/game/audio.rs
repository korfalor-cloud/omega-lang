/// Audio engine for game sound management.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AudioClip {
    pub name: String,
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: f64,
}

impl AudioClip {
    pub fn new(name: &str, data: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        let duration = data.len() as f64 / (sample_rate as f64 * channels as f64);
        Self {
            name: name.to_string(),
            data,
            sample_rate,
            channels,
            duration,
        }
    }

    pub fn silence(name: &str, duration: f64, sample_rate: u32) -> Self {
        let samples = (duration * sample_rate as f64) as usize;
        Self::new(name, vec![0.0; samples], sample_rate, 1)
    }

    pub fn sine_wave(name: &str, frequency: f64, duration: f64, sample_rate: u32) -> Self {
        let samples = (duration * sample_rate as f64) as usize;
        let data: Vec<f32> = (0..samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * frequency * t).sin() as f32
            })
            .collect();
        Self::new(name, data, sample_rate, 1)
    }

    pub fn square_wave(name: &str, frequency: f64, duration: f64, sample_rate: u32) -> Self {
        let samples = (duration * sample_rate as f64) as usize;
        let data: Vec<f32> = (0..samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let phase = (frequency * t) % 1.0;
                if phase < 0.5 { 1.0 } else { -1.0 }
            })
            .collect();
        Self::new(name, data, sample_rate, 1)
    }

    pub fn noise(name: &str, duration: f64, sample_rate: u32) -> Self {
        let samples = (duration * sample_rate as f64) as usize;
        let mut state: u32 = 12345;
        let data: Vec<f32> = (0..samples)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                ((state >> 16) as f32 / 32768.0) - 1.0
            })
            .collect();
        Self::new(name, data, sample_rate, 1)
    }
}

#[derive(Debug, Clone)]
pub struct AudioSource {
    pub clip_name: String,
    pub volume: f64,
    pub pitch: f64,
    pub looping: bool,
    pub playing: bool,
    pub position: f64,
}

impl AudioSource {
    pub fn new(clip_name: &str) -> Self {
        Self {
            clip_name: clip_name.to_string(),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            playing: false,
            position: 0.0,
        }
    }

    pub fn volume(mut self, volume: f64) -> Self {
        self.volume = volume.clamp(0.0, 1.0);
        self
    }

    pub fn pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch.clamp(0.1, 4.0);
        self
    }

    pub fn looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    pub fn play(&mut self) {
        self.playing = true;
        self.position = 0.0;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.position = 0.0;
    }
}

#[derive(Debug)]
pub struct AudioEngine {
    clips: HashMap<String, AudioClip>,
    sources: Vec<AudioSource>,
    master_volume: f64,
    muted: bool,
}

impl AudioEngine {
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
            sources: Vec::new(),
            master_volume: 1.0,
            muted: false,
        }
    }

    pub fn load_clip(&mut self, clip: AudioClip) {
        self.clips.insert(clip.name.clone(), clip);
    }

    pub fn create_source(&mut self, clip_name: &str) -> usize {
        let source = AudioSource::new(clip_name);
        let id = self.sources.len();
        self.sources.push(source);
        id
    }

    pub fn play(&mut self, source_id: usize) {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.play();
        }
    }

    pub fn pause(&mut self, source_id: usize) {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.pause();
        }
    }

    pub fn stop(&mut self, source_id: usize) {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.stop();
        }
    }

    pub fn set_volume(&mut self, source_id: usize, volume: f64) {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.volume = volume.clamp(0.0, 1.0);
        }
    }

    pub fn set_master_volume(&mut self, volume: f64) {
        self.master_volume = volume.clamp(0.0, 1.0);
    }

    pub fn mute(&mut self) {
        self.muted = true;
    }

    pub fn unmute(&mut self) {
        self.muted = false;
    }

    pub fn is_muted(&self) -> bool {
        self.muted
    }

    pub fn update(&mut self, dt: f64) {
        for source in &mut self.sources {
            if !source.playing {
                continue;
            }

            if let Some(clip) = self.clips.get(&source.clip_name) {
                source.position += dt * source.pitch;
                if source.position >= clip.duration {
                    if source.looping {
                        source.position -= clip.duration;
                    } else {
                        source.playing = false;
                        source.position = 0.0;
                    }
                }
            }
        }
    }

    pub fn playing_sources(&self) -> Vec<usize> {
        self.sources.iter().enumerate()
            .filter(|(_, s)| s.playing)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    pub fn get_clip(&self, name: &str) -> Option<&AudioClip> {
        self.clips.get(name)
    }

    pub fn play_one_shot(&mut self, clip_name: &str, volume: f64) {
        let mut source = AudioSource::new(clip_name).volume(volume);
        source.play();
        self.sources.push(source);
    }
}

/// Mixer for combining audio streams
pub struct Mixer {
    streams: Vec<Vec<f32>>,
    volumes: Vec<f64>,
}

impl Mixer {
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
            volumes: Vec::new(),
        }
    }

    pub fn add_stream(&mut self, data: Vec<f32>, volume: f64) {
        self.streams.push(data);
        self.volumes.push(volume);
    }

    pub fn mix(&self, output_len: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; output_len];

        for (stream, &volume) in self.streams.iter().zip(self.volumes.iter()) {
            for (i, sample) in stream.iter().take(output_len).enumerate() {
                output[i] += sample * volume as f32;
            }
        }

        // Clip to prevent distortion
        for sample in &mut output {
            *sample = sample.clamp(-1.0, 1.0);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_wave() {
        let clip = AudioClip::sine_wave("test", 440.0, 1.0, 44100);
        assert_eq!(clip.data.len(), 44100);
        assert!((clip.duration - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_engine() {
        let mut engine = AudioEngine::new();
        let clip = AudioClip::sine_wave("beep", 440.0, 0.5, 44100);
        engine.load_clip(clip);

        let source = engine.create_source("beep");
        engine.play(source);

        assert!(engine.playing_sources().contains(&source));
    }

    #[test]
    fn test_volume_control() {
        let mut engine = AudioEngine::new();
        engine.set_master_volume(0.5);
        assert_eq!(engine.master_volume, 0.5);
    }

    #[test]
    fn test_mixer() {
        let mut mixer = Mixer::new();
        mixer.add_stream(vec![0.5, 0.5, 0.5], 1.0);
        mixer.add_stream(vec![0.3, 0.3, 0.3], 0.5);

        let mixed = mixer.mix(3);
        assert!((mixed[0] - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_looping() {
        let mut engine = AudioEngine::new();
        let clip = AudioClip::sine_wave("loop", 440.0, 1.0, 44100);
        engine.load_clip(clip);

        let mut source = AudioSource::new("loop").looping(true);
        source.play();
        engine.sources.push(source);

        engine.update(1.5); // Past clip duration
        assert!(engine.sources[0].playing); // Should still be playing (looping)
    }
}
