//! Frame codec: length-prefixed typed frames over async byte streams.

use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Frame types carried over the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    /// JSON-RPC 2.0 request (client → daemon).
    Request = 0x01,
    /// JSON-RPC 2.0 response (daemon → client).
    Response = 0x02,
    /// Daemon-initiated event/notification (daemon → client).
    Event = 0x03,
    /// Automerge sync message (bidirectional, future).
    Sync = 0x04,
}

impl FrameType {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Request),
            0x02 => Some(Self::Response),
            0x03 => Some(Self::Event),
            0x04 => Some(Self::Sync),
            _ => None,
        }
    }
}

/// A typed frame with its payload.
#[derive(Debug, Clone)]
pub struct Frame {
    pub frame_type: FrameType,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn request(json: &[u8]) -> Self {
        Self {
            frame_type: FrameType::Request,
            payload: json.to_vec(),
        }
    }

    pub fn response(json: &[u8]) -> Self {
        Self {
            frame_type: FrameType::Response,
            payload: json.to_vec(),
        }
    }

    pub fn event(json: &[u8]) -> Self {
        Self {
            frame_type: FrameType::Event,
            payload: json.to_vec(),
        }
    }

    /// Parse the payload as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_slice(&self.payload)
    }
}

/// Maximum frame payload size (1 MB).
const MAX_FRAME_SIZE: u32 = 1_048_576;

/// Write a frame to an async writer.
pub async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, frame: &Frame) -> io::Result<()> {
    let payload_len = frame.payload.len() as u32 + 1; // +1 for frame type byte
    writer.write_all(&payload_len.to_be_bytes()).await?;
    writer.write_u8(frame.frame_type as u8).await?;
    writer.write_all(&frame.payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Read a frame from an async reader. Returns None on EOF.
pub async fn read_frame<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<Option<Frame>> {
    // Read length prefix
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    let total_len = u32::from_be_bytes(len_buf);
    if total_len == 0 || total_len > MAX_FRAME_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame too large: {} bytes", total_len),
        ));
    }

    // Read frame type byte
    let type_byte = reader.read_u8().await?;
    let frame_type = FrameType::from_byte(type_byte).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown frame type: 0x{:02x}", type_byte),
        )
    })?;

    // Read payload
    let payload_len = (total_len - 1) as usize;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload).await?;

    Ok(Some(Frame {
        frame_type,
        payload,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[tokio::test]
    async fn test_roundtrip() {
        let (mut client, mut server) = duplex(1024);

        let request = Frame::request(b"{\"method\":\"speak\"}");
        write_frame(&mut client, &request).await.unwrap();

        let received = read_frame(&mut server).await.unwrap().unwrap();
        assert_eq!(received.frame_type, FrameType::Request);
        assert_eq!(received.payload, b"{\"method\":\"speak\"}");
    }

    #[tokio::test]
    async fn test_multiple_frames() {
        let (mut client, mut server) = duplex(4096);

        let f1 = Frame::request(b"one");
        let f2 = Frame::response(b"two");
        let f3 = Frame::event(b"three");

        write_frame(&mut client, &f1).await.unwrap();
        write_frame(&mut client, &f2).await.unwrap();
        write_frame(&mut client, &f3).await.unwrap();

        let r1 = read_frame(&mut server).await.unwrap().unwrap();
        let r2 = read_frame(&mut server).await.unwrap().unwrap();
        let r3 = read_frame(&mut server).await.unwrap().unwrap();

        assert_eq!(r1.frame_type, FrameType::Request);
        assert_eq!(r2.frame_type, FrameType::Response);
        assert_eq!(r3.frame_type, FrameType::Event);
    }

    #[tokio::test]
    async fn test_eof_returns_none() {
        let (client, mut server) = duplex(1024);
        drop(client);
        let result = read_frame(&mut server).await.unwrap();
        assert!(result.is_none());
    }
}
