// Networking tools (basic UDP server/client for state sync).

use std::net::{UdpSocket, SocketAddr};
use std::thread;
use std::time::Duration;

pub fn available() -> bool { true }

pub struct Server {
    socket: UdpSocket,
}

impl Server {
    /// Bind to the given address (e.g., "127.0.0.1:0" for an ephemeral port).
    pub fn bind(addr: &str) -> Result<Self, String> {
        let sock = UdpSocket::bind(addr).map_err(|e| e.to_string())?;
        sock.set_nonblocking(true).map_err(|e| e.to_string())?;
        Ok(Server { socket: sock })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, String> {
        self.socket.local_addr().map_err(|e| e.to_string())
    }

    /// Try to receive a datagram (non-blocking). Returns sender and payload if available.
    pub fn try_recv(&self) -> Result<Option<(SocketAddr, Vec<u8>)>, String> {
        let mut buf = [0u8; 65536];
        match self.socket.recv_from(&mut buf) {
            Ok((n, src)) => Ok(Some((src, buf[..n].to_vec()))),
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn send_to(&self, dest: &SocketAddr, data: &[u8]) -> Result<(), String> {
        self.socket.send_to(data, dest).map_err(|e| e.to_string())?;
        Ok(())
    }
}

pub fn send_one(addr: &str, data: &[u8]) -> Result<(), String> {
    let sock = UdpSocket::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    sock.send_to(data, addr).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn udp_roundtrip() {
        let server = Server::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        // Spawn a sender thread
        let data = b"hello".to_vec();
        let dest = addr.clone();
        thread::spawn(move || {
            let _ = send_one(&dest.to_string(), &data);
        });
        let start = Instant::now();
        // wait up to 1s
        loop {
            if let Some((src, payload)) = server.try_recv().unwrap() {
                assert_eq!(payload, b"hello");
                break;
            }
            if start.elapsed() > Duration::from_secs(1) { panic!("timeout"); }
            thread::sleep(Duration::from_millis(10));
        }
    }
}
