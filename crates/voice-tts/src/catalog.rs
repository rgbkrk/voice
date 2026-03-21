//! Voice catalog — metadata for all known Kokoro voices.
//!
//! Ships a static catalog of every voice available in the Kokoro model,
//! with display name, language, gender, and quality grade. Also provides
//! runtime helpers to check which voices are cached locally via HF Hub.

use serde::Serialize;

/// Metadata for a single Kokoro voice.
#[derive(Debug, Clone, Serialize)]
pub struct VoiceInfo {
    /// Machine identifier, e.g. `"af_heart"`
    pub id: &'static str,
    /// Human-friendly display name, e.g. `"Heart"`
    pub name: &'static str,
    /// Language label, e.g. `"American English"`
    pub language: &'static str,
    /// `"Female"` or `"Male"`
    pub gender: &'static str,
    /// Quality grade from the Kokoro project (`"A"`, `"B-"`, `"C+"`, etc.)
    pub grade: &'static str,
    /// Special traits (e.g. `"❤️"`, `"🔥"`, `"🎧"`)
    pub traits: &'static str,
}

/// Whether a voice is embedded in the binary.
pub fn is_builtin(id: &str) -> bool {
    crate::builtin::BUILTIN_VOICES.contains(&id)
}

/// Check whether a voice is cached locally in the HF Hub cache.
///
/// This inspects the cache directory without downloading anything.
pub fn is_cached(id: &str, repo_id: Option<&str>) -> bool {
    let repo_id = repo_id.unwrap_or("prince-canuma/Kokoro-82M");
    let voice_path = format!("voices/{}.safetensors", id);

    let cache = hf_hub::Cache::default();
    let repo = cache.repo(hf_hub::Repo::model(repo_id.to_string()));
    repo.get(&voice_path).is_some()
}

/// Look up a voice by ID.
pub fn voice_info(id: &str) -> Option<&'static VoiceInfo> {
    ALL_VOICES.iter().find(|v| v.id == id)
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Derive a display name from a voice ID (e.g. `"af_heart"` → `"Heart"`).
pub fn display_name(id: &str) -> String {
    id.split('_')
        .skip(1) // skip language/gender prefix
        .map(|s| capitalize(s))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Full catalog ────────────────────────────────────────────────────────

/// Every known Kokoro voice with metadata.
pub static ALL_VOICES: &[VoiceInfo] = &[
    // ── American English (11F, 9M) ──────────────────────────────────
    VoiceInfo { id: "af_heart",   name: "Heart",   language: "American English", gender: "Female", grade: "A",  traits: "❤️" },
    VoiceInfo { id: "af_alloy",   name: "Alloy",   language: "American English", gender: "Female", grade: "C",  traits: "" },
    VoiceInfo { id: "af_aoede",   name: "Aoede",   language: "American English", gender: "Female", grade: "C+", traits: "" },
    VoiceInfo { id: "af_bella",   name: "Bella",   language: "American English", gender: "Female", grade: "A-", traits: "🔥" },
    VoiceInfo { id: "af_jessica", name: "Jessica", language: "American English", gender: "Female", grade: "D",  traits: "" },
    VoiceInfo { id: "af_kore",    name: "Kore",    language: "American English", gender: "Female", grade: "C+", traits: "" },
    VoiceInfo { id: "af_nicole",  name: "Nicole",  language: "American English", gender: "Female", grade: "B-", traits: "🎧" },
    VoiceInfo { id: "af_nova",    name: "Nova",    language: "American English", gender: "Female", grade: "C",  traits: "" },
    VoiceInfo { id: "af_river",   name: "River",   language: "American English", gender: "Female", grade: "D",  traits: "" },
    VoiceInfo { id: "af_sarah",   name: "Sarah",   language: "American English", gender: "Female", grade: "C+", traits: "" },
    VoiceInfo { id: "af_sky",     name: "Sky",     language: "American English", gender: "Female", grade: "C-", traits: "" },
    VoiceInfo { id: "am_adam",    name: "Adam",    language: "American English", gender: "Male",   grade: "F+", traits: "" },
    VoiceInfo { id: "am_echo",    name: "Echo",    language: "American English", gender: "Male",   grade: "D",  traits: "" },
    VoiceInfo { id: "am_eric",    name: "Eric",    language: "American English", gender: "Male",   grade: "D",  traits: "" },
    VoiceInfo { id: "am_fenrir",  name: "Fenrir",  language: "American English", gender: "Male",   grade: "C+", traits: "" },
    VoiceInfo { id: "am_liam",    name: "Liam",    language: "American English", gender: "Male",   grade: "D",  traits: "" },
    VoiceInfo { id: "am_michael", name: "Michael", language: "American English", gender: "Male",   grade: "C+", traits: "" },
    VoiceInfo { id: "am_onyx",    name: "Onyx",    language: "American English", gender: "Male",   grade: "D",  traits: "" },
    VoiceInfo { id: "am_puck",    name: "Puck",    language: "American English", gender: "Male",   grade: "C+", traits: "" },
    VoiceInfo { id: "am_santa",   name: "Santa",   language: "American English", gender: "Male",   grade: "D-", traits: "" },

    // ── British English (4F, 4M) ────────────────────────────────────
    VoiceInfo { id: "bf_alice",    name: "Alice",    language: "British English", gender: "Female", grade: "D",  traits: "" },
    VoiceInfo { id: "bf_emma",     name: "Emma",     language: "British English", gender: "Female", grade: "B-", traits: "" },
    VoiceInfo { id: "bf_isabella", name: "Isabella", language: "British English", gender: "Female", grade: "C",  traits: "" },
    VoiceInfo { id: "bf_lily",     name: "Lily",     language: "British English", gender: "Female", grade: "D",  traits: "" },
    VoiceInfo { id: "bm_daniel",   name: "Daniel",   language: "British English", gender: "Male",   grade: "D",  traits: "" },
    VoiceInfo { id: "bm_fable",    name: "Fable",    language: "British English", gender: "Male",   grade: "C",  traits: "" },
    VoiceInfo { id: "bm_george",   name: "George",   language: "British English", gender: "Male",   grade: "C",  traits: "" },
    VoiceInfo { id: "bm_lewis",    name: "Lewis",    language: "British English", gender: "Male",   grade: "D+", traits: "" },

    // ── Japanese (4F, 1M) ───────────────────────────────────────────
    VoiceInfo { id: "jf_alpha",      name: "Alpha",      language: "Japanese", gender: "Female", grade: "C+", traits: "" },
    VoiceInfo { id: "jf_gongitsune", name: "Gongitsune", language: "Japanese", gender: "Female", grade: "C",  traits: "" },
    VoiceInfo { id: "jf_nezumi",     name: "Nezumi",     language: "Japanese", gender: "Female", grade: "C-", traits: "" },
    VoiceInfo { id: "jf_tebukuro",   name: "Tebukuro",   language: "Japanese", gender: "Female", grade: "C",  traits: "" },
    VoiceInfo { id: "jm_kumo",       name: "Kumo",       language: "Japanese", gender: "Male",   grade: "C-", traits: "" },

    // ── Mandarin Chinese (4F, 4M) ───────────────────────────────────
    VoiceInfo { id: "zf_xiaobei",  name: "Xiaobei",  language: "Mandarin Chinese", gender: "Female", grade: "D", traits: "" },
    VoiceInfo { id: "zf_xiaoni",   name: "Xiaoni",   language: "Mandarin Chinese", gender: "Female", grade: "D", traits: "" },
    VoiceInfo { id: "zf_xiaoxiao", name: "Xiaoxiao", language: "Mandarin Chinese", gender: "Female", grade: "D", traits: "" },
    VoiceInfo { id: "zf_xiaoyi",   name: "Xiaoyi",   language: "Mandarin Chinese", gender: "Female", grade: "D", traits: "" },
    VoiceInfo { id: "zm_yunjian",  name: "Yunjian",  language: "Mandarin Chinese", gender: "Male",   grade: "D", traits: "" },
    VoiceInfo { id: "zm_yunxi",    name: "Yunxi",    language: "Mandarin Chinese", gender: "Male",   grade: "D", traits: "" },
    VoiceInfo { id: "zm_yunxia",   name: "Yunxia",   language: "Mandarin Chinese", gender: "Male",   grade: "D", traits: "" },
    VoiceInfo { id: "zm_yunyang",  name: "Yunyang",  language: "Mandarin Chinese", gender: "Male",   grade: "D", traits: "" },

    // ── Spanish (1F, 2M) ────────────────────────────────────────────
    VoiceInfo { id: "ef_dora",  name: "Dora",  language: "Spanish", gender: "Female", grade: "", traits: "" },
    VoiceInfo { id: "em_alex",  name: "Alex",  language: "Spanish", gender: "Male",   grade: "", traits: "" },
    VoiceInfo { id: "em_santa", name: "Santa", language: "Spanish", gender: "Male",   grade: "", traits: "" },

    // ── French (1F) ─────────────────────────────────────────────────
    VoiceInfo { id: "ff_siwis", name: "Siwis", language: "French", gender: "Female", grade: "B-", traits: "" },

    // ── Hindi (2F, 2M) ──────────────────────────────────────────────
    VoiceInfo { id: "hf_alpha", name: "Alpha", language: "Hindi", gender: "Female", grade: "C", traits: "" },
    VoiceInfo { id: "hf_beta",  name: "Beta",  language: "Hindi", gender: "Female", grade: "C", traits: "" },
    VoiceInfo { id: "hm_omega", name: "Omega", language: "Hindi", gender: "Male",   grade: "C", traits: "" },
    VoiceInfo { id: "hm_psi",   name: "Psi",   language: "Hindi", gender: "Male",   grade: "C", traits: "" },

    // ── Italian (1F, 1M) ────────────────────────────────────────────
    VoiceInfo { id: "if_sara",   name: "Sara",   language: "Italian", gender: "Female", grade: "C", traits: "" },
    VoiceInfo { id: "im_nicola", name: "Nicola", language: "Italian", gender: "Male",   grade: "C", traits: "" },

    // ── Brazilian Portuguese (1F, 2M) ───────────────────────────────
    VoiceInfo { id: "pf_dora",  name: "Dora",  language: "Brazilian Portuguese", gender: "Female", grade: "", traits: "" },
    VoiceInfo { id: "pm_alex",  name: "Alex",  language: "Brazilian Portuguese", gender: "Male",   grade: "", traits: "" },
    VoiceInfo { id: "pm_santa", name: "Santa", language: "Brazilian Portuguese", gender: "Male",   grade: "", traits: "" },
];
