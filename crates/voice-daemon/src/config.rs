//! Shared daemon configuration — voice, speed, etc.
//!
//! Accessible from both the socket handler (for set_voice/set_speed/list_voices)
//! and the worker (for reading current defaults).

use std::sync::Mutex;

pub struct DaemonConfig {
    pub engine: Mutex<String>,
    pub kokoro_voice_name: Mutex<String>,
    pub voxtral_voice_name: Mutex<String>,
    pub voxtral_model: Mutex<String>,
    pub speed: Mutex<f32>,
}

impl DaemonConfig {
    pub fn new() -> Self {
        Self {
            engine: Mutex::new("kokoro".to_string()),
            kokoro_voice_name: Mutex::new("af_heart".to_string()),
            voxtral_voice_name: Mutex::new("casual_male".to_string()),
            voxtral_model: Mutex::new(voice_voxtral::DEFAULT_REPO.to_string()),
            speed: Mutex::new(1.0),
        }
    }

    pub fn get_engine(&self) -> String {
        self.engine.lock().unwrap().clone()
    }

    pub fn set_engine(&self, engine: String) {
        *self.engine.lock().unwrap() = engine;
    }

    pub fn get_voice_name_for_engine(&self, engine: &str) -> String {
        if engine == "voxtral" {
            self.voxtral_voice_name.lock().unwrap().clone()
        } else {
            self.kokoro_voice_name.lock().unwrap().clone()
        }
    }

    pub fn set_voice_name_for_engine(&self, engine: &str, name: String) {
        if engine == "voxtral" {
            *self.voxtral_voice_name.lock().unwrap() = name;
        } else {
            *self.kokoro_voice_name.lock().unwrap() = name;
        }
    }

    pub fn get_voxtral_model(&self) -> String {
        self.voxtral_model.lock().unwrap().clone()
    }

    pub fn set_voxtral_model(&self, model: String) {
        *self.voxtral_model.lock().unwrap() = model;
    }

    pub fn get_speed(&self) -> f32 {
        *self.speed.lock().unwrap()
    }

    pub fn set_speed(&self, speed: f32) {
        *self.speed.lock().unwrap() = speed;
    }
}
