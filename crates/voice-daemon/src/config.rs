//! Shared daemon configuration — voice, speed, etc.
//!
//! Accessible from both the socket handler (for set_voice/set_speed/list_voices)
//! and the worker (for reading current defaults).

use std::sync::Mutex;

pub struct DaemonConfig {
    pub voice_name: Mutex<String>,
    pub speed: Mutex<f32>,
}

impl DaemonConfig {
    pub fn new() -> Self {
        Self {
            voice_name: Mutex::new("af_heart".to_string()),
            speed: Mutex::new(1.0),
        }
    }

    pub fn get_voice_name(&self) -> String {
        self.voice_name.lock().unwrap().clone()
    }

    pub fn set_voice_name(&self, name: String) {
        *self.voice_name.lock().unwrap() = name;
    }

    pub fn get_speed(&self) -> f32 {
        *self.speed.lock().unwrap()
    }

    pub fn set_speed(&self, speed: f32) {
        *self.speed.lock().unwrap() = speed;
    }
}
