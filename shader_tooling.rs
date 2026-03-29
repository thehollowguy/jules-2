// Shader tooling (minimal local compiler stub).
// Validates basic syntax and writes a pseudo-compiled file to `shaders/`.

use std::fs;
use std::path::Path;

pub fn available() -> bool { true }

fn basic_validate(src: &str) -> Result<(), String> {
    if !src.contains("main") {
        return Err("shader missing 'main'".into());
    }
    let open = src.matches('{').count();
    let close = src.matches('}').count();
    if open != close {
        return Err("unbalanced braces".into());
    }
    Ok(())
}

pub fn compile_shader(name: &str, source: &str) -> Result<String, String> {
    basic_validate(source)?;
    fs::create_dir_all("shaders").map_err(|e| e.to_string())?;
    let out = format!("shaders/{}.spv", name);
    // Pseudo-compile: write the source prefixed with a small header.
    let mut buf = Vec::new();
    buf.extend_from_slice(b"SPV-STUB\n");
    buf.extend_from_slice(source.as_bytes());
    fs::write(&out, buf).map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_basic() {
        let src = "void main() { }";
        let out = compile_shader("test", src).unwrap();
        assert!(out.ends_with(".spv"));
        let _ = fs::remove_file(out);
    }
}
