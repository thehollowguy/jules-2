// Frame-by-frame debugger scaffold for Jules.
// Minimal, non-invasive API intended for future expansion into a full
// frame debugger with history, inspection, and rewind.

/// Is the frame debugger scaffold compiled in?
pub fn available() -> bool { true }

/// Step a single frame (stub). Real implementation will accept the runtime
/// world and scheduler handles to advance one tick and return diagnostics.
pub fn step_frame() -> &'static str {
    "frame stepped (stub)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_available() { assert!(available()); }

    #[test]
    fn step_frame_stub() { assert_eq!(step_frame(), "frame stepped (stub)"); }
}
