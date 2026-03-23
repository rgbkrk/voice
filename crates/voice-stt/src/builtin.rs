//! Built-in model metadata and defaults.

/// Default HuggingFace repo ID for STT.
/// distil-medium.en: English-only, fast, good accuracy.
pub const DEFAULT_MODEL_REPO: &str = "distil-whisper/distil-medium.en";

/// Known model repo IDs and whether they are multilingual.
pub fn is_multilingual(repo_id: &str) -> bool {
    // English-only models
    let en_only = [
        "openai/whisper-tiny.en",
        "openai/whisper-base.en",
        "openai/whisper-small.en",
        "openai/whisper-medium.en",
        "distil-whisper/distil-medium.en",
    ];
    !en_only.contains(&repo_id)
}
