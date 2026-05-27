/// Music theory: notes, scales, chords, progressions, rhythm.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Note {
    C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B,
}

impl Note {
    pub fn from_midi(midi: u8) -> Self {
        match midi % 12 {
            0 => Note::C, 1 => Note::Cs, 2 => Note::D, 3 => Note::Ds,
            4 => Note::E, 5 => Note::F, 6 => Note::Fs, 7 => Note::G,
            8 => Note::Gs, 9 => Note::A, 10 => Note::As, 11 => Note::B,
            _ => unreachable!(),
        }
    }

    pub fn to_midi(&self, octave: i8) -> u8 {
        let base = match self {
            Note::C => 0, Note::Cs => 1, Note::D => 2, Note::Ds => 3,
            Note::E => 4, Note::F => 5, Note::Fs => 6, Note::G => 7,
            Note::Gs => 8, Note::A => 9, Note::As => 10, Note::B => 11,
        };
        ((octave + 1) * 12 + base) as u8
    }

    pub fn frequency(&self, octave: i8) -> f64 {
        let midi = self.to_midi(octave) as f64;
        440.0 * 2.0_f64.powf((midi - 69.0) / 12.0)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Note::C => "C", Note::Cs => "C#", Note::D => "D", Note::Ds => "D#",
            Note::E => "E", Note::F => "F", Note::Fs => "F#", Note::G => "G",
            Note::Gs => "G#", Note::A => "A", Note::As => "A#", Note::B => "B",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "C" => Some(Note::C), "C#" | "DB" => Some(Note::Cs),
            "D" => Some(Note::D), "D#" | "EB" => Some(Note::Ds),
            "E" => Some(Note::E), "F" => Some(Note::F),
            "F#" | "GB" => Some(Note::Fs), "G" => Some(Note::G),
            "G#" | "AB" => Some(Note::Gs), "A" => Some(Note::A),
            "A#" | "BB" => Some(Note::As), "B" => Some(Note::B),
            _ => None,
        }
    }

    pub fn interval(self, semitones: u8) -> Self {
        let base = match self {
            Note::C => 0, Note::Cs => 1, Note::D => 2, Note::Ds => 3,
            Note::E => 4, Note::F => 5, Note::Fs => 6, Note::G => 7,
            Note::Gs => 8, Note::A => 9, Note::As => 10, Note::B => 11,
        };
        Self::from_midi((base + semitones) % 12)
    }

    pub fn all() -> [Note; 12] {
        [Note::C, Note::Cs, Note::D, Note::Ds, Note::E, Note::F,
         Note::Fs, Note::G, Note::Gs, Note::A, Note::As, Note::B]
    }

    pub fn sharp(&self) -> Option<Self> {
        Some(Self::from_midi((self.to_midi(0) + 1) % 12))
    }

    pub fn flat(&self) -> Option<Self> {
        Some(Self::from_midi((self.to_midi(0) + 11) % 12))
    }
}

/// Scale patterns (intervals from root).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleType {
    Major,
    Minor,
    HarmonicMinor,
    MelodicMinor,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Aeolian,
    Locrian,
    PentatonicMajor,
    PentatonicMinor,
    Blues,
    WholeTone,
    Chromatic,
    Diminished,
}

impl ScaleType {
    pub fn intervals(&self) -> &[u8] {
        match self {
            ScaleType::Major => &[0, 2, 4, 5, 7, 9, 11],
            ScaleType::Minor => &[0, 2, 3, 5, 7, 8, 10],
            ScaleType::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            ScaleType::MelodicMinor => &[0, 2, 3, 5, 7, 9, 11],
            ScaleType::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            ScaleType::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
            ScaleType::Lydian => &[0, 2, 4, 6, 7, 9, 11],
            ScaleType::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            ScaleType::Aeolian => &[0, 2, 3, 5, 7, 8, 10],
            ScaleType::Locrian => &[0, 1, 3, 5, 6, 8, 10],
            ScaleType::PentatonicMajor => &[0, 2, 4, 7, 9],
            ScaleType::PentatonicMinor => &[0, 3, 5, 7, 10],
            ScaleType::Blues => &[0, 3, 5, 6, 7, 10],
            ScaleType::WholeTone => &[0, 2, 4, 6, 8, 10],
            ScaleType::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ScaleType::Diminished => &[0, 2, 3, 5, 6, 8, 9, 11],
        }
    }
}

/// A scale with a root note and pattern.
#[derive(Debug, Clone)]
pub struct Scale {
    pub root: Note,
    pub scale_type: ScaleType,
    pub notes: Vec<Note>,
}

impl Scale {
    pub fn new(root: Note, scale_type: ScaleType) -> Self {
        let notes: Vec<Note> = scale_type.intervals().iter()
            .map(|&interval| root.interval(interval))
            .collect();
        Self { root, scale_type, notes }
    }

    pub fn notes(&self) -> &[Note] {
        &self.notes
    }

    pub fn contains(&self, note: Note) -> bool {
        self.notes.contains(&note)
    }

    pub fn degree(&self, degree: usize) -> Option<Note> {
        if degree == 0 || degree > self.notes.len() {
            return None;
        }
        Some(self.notes[(degree - 1) % self.notes.len()])
    }

    /// MIDI notes in a given octave range.
    pub fn midi_range(&self, start_octave: i8, end_octave: i8) -> Vec<u8> {
        let mut midis = Vec::new();
        for octave in start_octave..=end_octave {
            for note in &self.notes {
                let midi = note.to_midi(octave);
                if midi <= 127 {
                    midis.push(midi);
                }
            }
        }
        midis
    }
}

/// Chord types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChordType {
    Major,
    Minor,
    Diminished,
    Augmented,
    Major7,
    Minor7,
    Dominant7,
    Diminished7,
    HalfDiminished7,
    MinorMajor7,
    Sus2,
    Sus4,
    Add9,
    Major9,
    Minor9,
    Power, // perfect 5th only
}

impl ChordType {
    pub fn intervals(&self) -> &[u8] {
        match self {
            ChordType::Major => &[0, 4, 7],
            ChordType::Minor => &[0, 3, 7],
            ChordType::Diminished => &[0, 3, 6],
            ChordType::Augmented => &[0, 4, 8],
            ChordType::Major7 => &[0, 4, 7, 11],
            ChordType::Minor7 => &[0, 3, 7, 10],
            ChordType::Dominant7 => &[0, 4, 7, 10],
            ChordType::Diminished7 => &[0, 3, 6, 9],
            ChordType::HalfDiminished7 => &[0, 3, 6, 10],
            ChordType::MinorMajor7 => &[0, 3, 7, 11],
            ChordType::Sus2 => &[0, 2, 7],
            ChordType::Sus4 => &[0, 5, 7],
            ChordType::Add9 => &[0, 4, 7, 14],
            ChordType::Major9 => &[0, 4, 7, 11, 14],
            ChordType::Minor9 => &[0, 3, 7, 10, 14],
            ChordType::Power => &[0, 7],
        }
    }
}

/// A chord with a root and type.
#[derive(Debug, Clone)]
pub struct Chord {
    pub root: Note,
    pub chord_type: ChordType,
    pub notes: Vec<Note>,
    pub inversion: usize,
}

impl Chord {
    pub fn new(root: Note, chord_type: ChordType) -> Self {
        let notes: Vec<Note> = chord_type.intervals().iter()
            .map(|&interval| root.interval(interval))
            .collect();
        Self { root, chord_type, notes, inversion: 0 }
    }

    pub fn with_inversion(mut self, inversion: usize) -> Self {
        self.inversion = inversion % self.notes.len();
        self
    }

    pub fn notes(&self) -> Vec<Note> {
        let mut result = self.notes.clone();
        for _ in 0..self.inversion {
            let first = result.remove(0);
            result.push(first);
        }
        result
    }

    pub fn midi_notes(&self, octave: i8) -> Vec<u8> {
        self.notes().iter().enumerate().map(|(i, note)| {
            let mut midi = note.to_midi(octave);
            if i < self.inversion {
                midi += 12;
            }
            midi
        }).collect()
    }

    pub fn name(&self) -> String {
        let type_str = match self.chord_type {
            ChordType::Major => "",
            ChordType::Minor => "m",
            ChordType::Diminished => "dim",
            ChordType::Augmented => "aug",
            ChordType::Major7 => "maj7",
            ChordType::Minor7 => "m7",
            ChordType::Dominant7 => "7",
            ChordType::Diminished7 => "dim7",
            ChordType::HalfDiminished7 => "m7b5",
            ChordType::MinorMajor7 => "mMaj7",
            ChordType::Sus2 => "sus2",
            ChordType::Sus4 => "sus4",
            ChordType::Add9 => "add9",
            ChordType::Major9 => "maj9",
            ChordType::Minor9 => "m9",
            ChordType::Power => "5",
        };
        format!("{}{}", self.root.name(), type_str)
    }

    /// Identify chord from a set of notes.
    pub fn identify(notes: &[Note]) -> Vec<Self> {
        if notes.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::new();
        for root in &Note::all() {
            for chord_type in &[
                ChordType::Major, ChordType::Minor, ChordType::Diminished,
                ChordType::Augmented, ChordType::Major7, ChordType::Minor7,
                ChordType::Dominant7, ChordType::Sus2, ChordType::Sus4,
            ] {
                let chord = Chord::new(*root, *chord_type);
                let chord_notes: Vec<Note> = chord.notes.iter().take(3).copied().collect();
                if notes.iter().all(|n| chord_notes.contains(n)) {
                    results.push(chord);
                }
            }
        }
        results
    }
}

/// Common chord progressions.
pub fn chord_progression(key: Note, scale_type: ScaleType, degrees: &[usize]) -> Vec<Chord> {
    let scale = Scale::new(key, scale_type);
    degrees.iter().filter_map(|&degree| {
        if degree == 0 || degree > scale.notes.len() {
            return None;
        }
        let root = scale.notes[degree - 1];
        let chord_type = match scale_type {
            ScaleType::Major => match degree {
                1 | 4 | 5 => ChordType::Major,
                2 | 3 | 6 => ChordType::Minor,
                7 => ChordType::Diminished,
                _ => ChordType::Major,
            },
            ScaleType::Minor => match degree {
                1 | 4 | 5 => ChordType::Minor,
                3 | 6 | 7 => ChordType::Major,
                2 => ChordType::Diminished,
                _ => ChordType::Minor,
            },
            _ => ChordType::Major,
        };
        Some(Chord::new(root, chord_type))
    }).collect()
}

/// I-IV-V-I progression in the given key.
pub fn I_IV_V_I(key: Note) -> Vec<Chord> {
    chord_progression(key, ScaleType::Major, &[1, 4, 5, 1])
}

/// ii-V-I jazz progression.
pub fn ii_V_I(key: Note) -> Vec<Chord> {
    chord_progression(key, ScaleType::Major, &[2, 5, 1])
}

/// 12-bar blues progression.
pub fn blues_12_bar(key: Note) -> Vec<Chord> {
    chord_progression(key, ScaleType::Major, &[
        1, 1, 1, 1,
        4, 4, 1, 1,
        5, 4, 1, 5,
    ])
}

/// Note duration.
#[derive(Debug, Clone, Copy)]
pub enum Duration {
    Whole,
    Half,
    Quarter,
    Eighth,
    Sixteenth,
    Dotted(Box<Duration>),
    Triplet(Box<Duration>),
}

impl Duration {
    pub fn beats(&self) -> f64 {
        match self {
            Duration::Whole => 4.0,
            Duration::Half => 2.0,
            Duration::Quarter => 1.0,
            Duration::Eighth => 0.5,
            Duration::Sixteenth => 0.25,
            Duration::Dotted(d) => d.beats() * 1.5,
            Duration::Triplet(d) => d.beats() * 2.0 / 3.0,
        }
    }

    pub fn seconds(&self, bpm: f64) -> f64 {
        self.beats() * 60.0 / bpm
    }
}

/// Musical note event.
#[derive(Debug, Clone)]
pub struct NoteEvent {
    pub note: Note,
    pub octave: i8,
    pub duration: Duration,
    pub velocity: u8, // 0-127
}

impl NoteEvent {
    pub fn new(note: Note, octave: i8, duration: Duration) -> Self {
        Self { note, octave, duration, velocity: 100 }
    }

    pub fn with_velocity(mut self, velocity: u8) -> Self {
        self.velocity = velocity;
        self
    }

    pub fn midi(&self) -> u8 {
        self.note.to_midi(self.octave)
    }

    pub fn frequency(&self) -> f64 {
        self.note.frequency(self.octave)
    }

    pub fn duration_seconds(&self, bpm: f64) -> f64 {
        self.duration.seconds(bpm)
    }
}

/// Time signature.
#[derive(Debug, Clone, Copy)]
pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
}

impl TimeSignature {
    pub fn new(numerator: u8, denominator: u8) -> Self {
        Self { numerator, denominator }
    }

    pub fn beats_per_measure(&self) -> u8 {
        self.numerator
    }

    pub fn beat_value(&self) -> f64 {
        4.0 / self.denominator as f64
    }
}

/// Simple melody builder.
pub struct Melody {
    notes: Vec<NoteEvent>,
    bpm: f64,
    time_sig: TimeSignature,
}

impl Melody {
    pub fn new(bpm: f64) -> Self {
        Self {
            notes: Vec::new(),
            bpm,
            time_sig: TimeSignature::new(4, 4),
        }
    }

    pub fn with_time_signature(mut self, ts: TimeSignature) -> Self {
        self.time_sig = ts;
        self
    }

    pub fn add_note(&mut self, note: NoteEvent) {
        self.notes.push(note);
    }

    pub fn add_notes(&mut self, notes: &[NoteEvent]) {
        self.notes.extend_from_slice(notes);
    }

    pub fn notes(&self) -> &[NoteEvent] {
        &self.notes
    }

    pub fn total_duration(&self) -> f64 {
        self.notes.iter().map(|n| n.duration_seconds(self.bpm)).sum()
    }

    pub fn total_beats(&self) -> f64 {
        self.notes.iter().map(|n| n.duration.beats()).sum()
    }

    pub fn note_count(&self) -> usize {
        self.notes.len()
    }

    pub fn bpm(&self) -> f64 {
        self.bpm
    }

    /// MIDI note sequence with timing.
    pub fn to_midi_sequence(&self) -> Vec<(u8, f64, f64)> {
        let mut time = 0.0;
        let mut result = Vec::new();
        for note in &self.notes {
            let duration = note.duration_seconds(self.bpm);
            result.push((note.midi(), time, duration));
            time += duration;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_frequency() {
        let a4 = Note::A.frequency(4);
        assert!((a4 - 440.0).abs() < 0.01);

        let c4 = Note::C.frequency(4);
        assert!((c4 - 261.63).abs() < 0.1);
    }

    #[test]
    fn test_scale() {
        let c_major = Scale::new(Note::C, ScaleType::Major);
        assert_eq!(c_major.notes, vec![Note::C, Note::D, Note::E, Note::F, Note::G, Note::A, Note::B]);

        let a_minor = Scale::new(Note::A, ScaleType::Minor);
        assert_eq!(a_minor.notes, vec![Note::A, Note::B, Note::C, Note::D, Note::E, Note::F, Note::G]);
    }

    #[test]
    fn test_chord() {
        let c_major = Chord::new(Note::C, ChordType::Major);
        assert_eq!(c_major.name(), "C");
        assert_eq!(c_major.notes, vec![Note::C, Note::E, Note::G]);

        let a_minor = Chord::new(Note::A, ChordType::Minor);
        assert_eq!(a_minor.name(), "Am");
    }

    #[test]
    fn test_progression() {
        let prog = I_IV_V_I(Note::C);
        assert_eq!(prog.len(), 4);
        assert_eq!(prog[0].name(), "C");
        assert_eq!(prog[1].name(), "F");
        assert_eq!(prog[2].name(), "G");
        assert_eq!(prog[3].name(), "C");
    }
}
