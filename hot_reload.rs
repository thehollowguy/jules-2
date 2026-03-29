// Hot-reload scaffold.
// Initial safe API: request reload for named module or asset; implement
// single-threaded reload guarantees in v0 and add concurrency later.

pub fn available() -> bool { true }

pub fn reload_module(_name: &str) -> Result<(), String> { Ok(()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_reload_stub() { assert!(available()); }
}
