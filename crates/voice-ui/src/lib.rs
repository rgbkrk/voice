//! Embedded assets for the Voice UI web surface.
//!
//! Build the frontend first with `npm run build` from `crates/voice-ui`.
//! Cargo then embeds the generated `dist` files into Rust binaries that depend
//! on this crate.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddedAsset {
    pub path: &'static str,
    pub mime: &'static str,
    pub bytes: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/voice_ui_assets.rs"));

pub fn assets() -> &'static [EmbeddedAsset] {
    ASSETS
}

pub fn get(path: impl AsRef<str>) -> Option<&'static EmbeddedAsset> {
    let normalized = normalize_path(path.as_ref());
    ASSETS.iter().find(|asset| asset.path == normalized)
}

pub fn index_html() -> Option<&'static EmbeddedAsset> {
    get("index.html")
}

fn normalize_path(path: &str) -> String {
    let trimmed = path.trim_start_matches('/');
    if trimmed.is_empty() {
        "index.html".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_path_maps_to_index() {
        assert_eq!(normalize_path("/"), "index.html");
        assert_eq!(normalize_path(""), "index.html");
    }

    #[test]
    fn built_dist_contains_index_when_assets_exist() {
        if !assets().is_empty() {
            assert!(index_html().is_some());
            assert!(get("/").is_some());
        }
    }
}
