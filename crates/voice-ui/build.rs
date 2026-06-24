use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=VOICE_UI_DIST");
    println!("cargo:rerun-if-changed=dist");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let dist_dir = env::var_os("VOICE_UI_DIST")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("dist"));

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let generated = out_dir.join("voice_ui_assets.rs");

    let assets = if dist_dir.exists() {
        match collect_assets(&dist_dir) {
            Ok(assets) => assets,
            Err(error) => panic!("failed to collect voice-ui assets: {error}"),
        }
    } else {
        Vec::new()
    };

    let mut source = String::from("pub static ASSETS: &[EmbeddedAsset] = &[\n");
    for asset in assets {
        source.push_str(&format!(
            "    EmbeddedAsset {{ path: {:?}, mime: {:?}, bytes: include_bytes!({:?}) }},\n",
            asset.path,
            mime_for(&asset.path),
            asset.absolute_path.display().to_string()
        ));
    }
    source.push_str("];\n");

    fs::write(generated, source).expect("failed to write generated voice-ui assets");
}

struct Asset {
    path: String,
    absolute_path: PathBuf,
}

fn collect_assets(dist_dir: &Path) -> io::Result<Vec<Asset>> {
    let mut assets = Vec::new();
    collect_dir(dist_dir, dist_dir, &mut assets)?;
    assets.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(assets)
}

fn collect_dir(root: &Path, dir: &Path, assets: &mut Vec<Asset>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_dir(root, &path, assets)?;
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .expect("asset path should be inside root")
                .to_string_lossy()
                .replace('\\', "/");
            assets.push(Asset {
                path: relative,
                absolute_path: path.canonicalize()?,
            });
        }
    }
    Ok(())
}

fn mime_for(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}
