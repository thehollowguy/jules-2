// Visual scene editor scaffold.
// Provides a minimal API surface for launching/communicating with an external
// editor or embedding a future UI. This is a placeholder for the real editor.

pub fn available() -> bool { true }

pub fn launch_editor() -> &'static str { "scene editor launched (stub)" }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_stub() { assert!(available()); }

    #[test]
    fn launch_stub() { assert_eq!(launch_editor(), "scene editor launched (stub)"); }
}
