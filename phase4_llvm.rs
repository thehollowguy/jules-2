//! Phase 4 LLVM integration module.
//!
//! This module exposes runtime capability checks and tuning metadata for the
//! optional LLVM-backed optimization stage.

use std::process::Command;

/// Returns whether the optional Phase 4 LLVM backend is available.
#[must_use]
pub fn is_available() -> bool {
    if !cfg!(feature = "phase4-llvm") {
        return false;
    }
    llvm_version().is_some()
}

/// Returns the detected `llvm-config --version` string when LLVM is available.
#[must_use]
pub fn llvm_version() -> Option<String> {
    let output = Command::new("llvm-config").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let v = String::from_utf8(output.stdout).ok()?;
    let trimmed = v.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Recommended optimization level for LLVM emission.
///
/// Mirrors Rust's release profile defaults unless `JULES_LLVM_OPT_LEVEL` is set.
#[must_use]
pub fn recommended_opt_level() -> u8 {
    std::env::var("JULES_LLVM_OPT_LEVEL")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .map(|v| v.min(3))
        .unwrap_or(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_level_in_range() {
        let lvl = recommended_opt_level();
        assert!(lvl <= 3);
    }

    #[test]
    fn availability_matches_probe() {
        if cfg!(feature = "phase4-llvm") {
            assert_eq!(is_available(), llvm_version().is_some());
        } else {
            assert!(!is_available());
        }
    }
}
