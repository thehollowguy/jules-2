// Asset importer pipeline (minimal implementation).
// Copies files into `assets/` and maintains a JSON manifest.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct AssetEntry {
    original: String,
    stored: String,
    size: u64,
    mtime: u64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct Manifest {
    entries: Vec<AssetEntry>,
}

const ASSET_DIR: &str = "assets";
const MANIFEST_FILE: &str = "assets/manifest.json";

pub fn available() -> bool { true }

pub fn import_asset(path: &str) -> Result<String, String> {
    let src = Path::new(path);
    if !src.exists() {
        return Err("source file not found".into());
    }
    fs::create_dir_all(ASSET_DIR).map_err(|e| e.to_string())?;
    let file_name = src.file_name().ok_or("invalid file name")?;
    let dest = Path::new(ASSET_DIR).join(file_name);
    fs::copy(src, &dest).map_err(|e| e.to_string())?;
    let meta = fs::metadata(&dest).map_err(|e| e.to_string())?;
    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let entry = AssetEntry {
        original: path.to_string(),
        stored: dest.to_string_lossy().to_string(),
        size: meta.len(),
        mtime,
    };
    // update manifest
    let mut manifest = if Path::new(MANIFEST_FILE).exists() {
        let txt = fs::read_to_string(MANIFEST_FILE).map_err(|e| e.to_string())?;
        serde_json::from_str(&txt).unwrap_or_default()
    } else { Manifest::default() };
    manifest.entries.push(entry);
    let txt = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::write(MANIFEST_FILE, txt).map_err(|e| e.to_string())?;
    Ok(dest.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    #[test]
    fn import_roundtrip() {
        let tmp = "test_asset.tmp";
        let mut f = File::create(tmp).unwrap();
        use std::io::Write;
        writeln!(f, "hello").unwrap();
        let res = import_asset(tmp).unwrap();
        assert!(res.contains("assets"));
        let _ = fs::remove_file(tmp);
        let _ = fs::remove_file(&res);
        let _ = fs::remove_file(MANIFEST_FILE);
    }
}
