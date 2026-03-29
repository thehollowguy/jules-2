// Profiling tools scaffold.
// Provide scope timers and a simple exporter interface; real work will add
// low-overhead counters and integration with perf/flamegraph.

pub fn available() -> bool { true }

pub fn start_scope(_name: &str) -> &'static str { "scope-started (stub)" }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_stub() { assert!(available()); }
}
