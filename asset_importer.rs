// Asset importer pipeline scaffold.
// Minimal CLI-friendly helpers to convert and register assets into engine-ready
// blobs. Real implementation will add format-specific importers (png, wav, gltf).

pub fn available() -> bool { true }

pub fn import_asset(path: &str) -> Result<String, String> {
    Ok(format!("imported: {} (stub)", path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_stub() { assert!(import_asset("foo.png").is_ok()); }
}
