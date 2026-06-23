/// Metadata for a preset Voxtral voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresetVoice {
    pub id: &'static str,
    pub display_name: &'static str,
    pub language: &'static str,
    pub gender: &'static str,
}

/// Official preset voice IDs exposed by `params.json`.
pub const VOXTRAL_PRESET_VOICES: &[PresetVoice] = &[
    PresetVoice {
        id: "casual_female",
        display_name: "Casual Female",
        language: "English",
        gender: "Female",
    },
    PresetVoice {
        id: "casual_male",
        display_name: "Casual Male",
        language: "English",
        gender: "Male",
    },
    PresetVoice {
        id: "cheerful_female",
        display_name: "Cheerful Female",
        language: "English",
        gender: "Female",
    },
    PresetVoice {
        id: "neutral_female",
        display_name: "Neutral Female",
        language: "English",
        gender: "Female",
    },
    PresetVoice {
        id: "neutral_male",
        display_name: "Neutral Male",
        language: "English",
        gender: "Male",
    },
    PresetVoice {
        id: "pt_male",
        display_name: "Portuguese Male",
        language: "Portuguese",
        gender: "Male",
    },
    PresetVoice {
        id: "pt_female",
        display_name: "Portuguese Female",
        language: "Portuguese",
        gender: "Female",
    },
    PresetVoice {
        id: "nl_male",
        display_name: "Dutch Male",
        language: "Dutch",
        gender: "Male",
    },
    PresetVoice {
        id: "nl_female",
        display_name: "Dutch Female",
        language: "Dutch",
        gender: "Female",
    },
    PresetVoice {
        id: "it_male",
        display_name: "Italian Male",
        language: "Italian",
        gender: "Male",
    },
    PresetVoice {
        id: "it_female",
        display_name: "Italian Female",
        language: "Italian",
        gender: "Female",
    },
    PresetVoice {
        id: "fr_male",
        display_name: "French Male",
        language: "French",
        gender: "Male",
    },
    PresetVoice {
        id: "fr_female",
        display_name: "French Female",
        language: "French",
        gender: "Female",
    },
    PresetVoice {
        id: "es_male",
        display_name: "Spanish Male",
        language: "Spanish",
        gender: "Male",
    },
    PresetVoice {
        id: "es_female",
        display_name: "Spanish Female",
        language: "Spanish",
        gender: "Female",
    },
    PresetVoice {
        id: "de_male",
        display_name: "German Male",
        language: "German",
        gender: "Male",
    },
    PresetVoice {
        id: "de_female",
        display_name: "German Female",
        language: "German",
        gender: "Female",
    },
    PresetVoice {
        id: "ar_male",
        display_name: "Arabic Male",
        language: "Arabic",
        gender: "Male",
    },
    PresetVoice {
        id: "hi_male",
        display_name: "Hindi Male",
        language: "Hindi",
        gender: "Male",
    },
    PresetVoice {
        id: "hi_female",
        display_name: "Hindi Female",
        language: "Hindi",
        gender: "Female",
    },
];

pub fn get_preset_voice(id: &str) -> Option<&'static PresetVoice> {
    VOXTRAL_PRESET_VOICES.iter().find(|voice| voice.id == id)
}
