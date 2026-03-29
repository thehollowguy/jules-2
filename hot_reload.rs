// Hot-reload implementation using simple mtime polling.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, Duration};
use std::thread;
use std::sync::{Arc, Mutex};
use std::fs;

type Callback = Box<dyn Fn(&PathBuf) + Send + Sync + 'static>;

#[derive(Clone)]
pub struct Watcher {
    watched: Arc<Mutex<HashMap<PathBuf, SystemTime>>>,
    callbacks: Arc<Mutex<Vec<Callback>>>,
}

impl Watcher {
    pub fn new() -> Self {
        Watcher { watched: Arc::new(Mutex::new(HashMap::new())), callbacks: Arc::new(Mutex::new(Vec::new())) }
    }

    pub fn watch<P: Into<PathBuf>>(&self, p: P) {
        let p = p.into();
        let mtime = fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
        self.watched.lock().unwrap().insert(p, mtime);
    }

    pub fn on_change<F: Fn(&PathBuf) + Send + Sync + 'static>(&self, f: F) {
        self.callbacks.lock().unwrap().push(Box::new(f));
    }

    /// Poll once for changes and invoke callbacks.
    pub fn poll_once(&self) {
        let mut to_call: Vec<PathBuf> = Vec::new();
        let mut w = self.watched.lock().unwrap();
        for (p, old) in w.iter_mut() {
            if let Ok(meta) = fs::metadata(p) {
                if let Ok(modified) = meta.modified() {
                    if modified > *old {
                        *old = modified;
                        to_call.push(p.clone());
                    }
                }
            }
        }
        drop(w);
        if !to_call.is_empty() {
            let cbs = self.callbacks.lock().unwrap().clone();
            for p in to_call {
                for cb in &cbs {
                    cb(&p);
                }
            }
        }
    }

    /// Start a background thread that polls every `interval` and fires callbacks.
    pub fn start_polling(&self, interval: Duration) {
        let s = self.clone();
        thread::spawn(move || loop {
            s.poll_once();
            thread::sleep(interval);
        });
    }
}

pub fn available() -> bool { true }

pub fn reload_module(_name: &str) -> Result<(), String> { Ok(()) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::time::Duration;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn watcher_triggers_on_change() {
        let tmp = "hotreload_test.tmp";
        let mut f = File::create(tmp).unwrap();
        writeln!(f, "hello").unwrap();
        let w = Watcher::new();
        w.watch(tmp);
        let called = Arc::new(Mutex::new(false));
        let c2 = called.clone();
        w.on_change(move |_p| {
            let mut g = c2.lock().unwrap(); *g = true;
        });
        w.start_polling(Duration::from_millis(50));
        // modify file
        thread::sleep(Duration::from_millis(20));
        let mut f = File::create(tmp).unwrap(); writeln!(f, "world").unwrap();
        // wait for callback
        thread::sleep(Duration::from_millis(200));
        assert!(*called.lock().unwrap());
        let _ = fs::remove_file(tmp);
    }
}
