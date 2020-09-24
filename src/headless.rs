use itertools::Itertools;

use crate::{
    common::film::Film,
    common::Camera,
    pathtracer::{integrator::PathIntegrator, RenderScene},
};
use std::ffi::CString;
use std::net::TcpStream;
use std::net::UdpSocket;
use std::{io::Write, sync::atomic::AtomicBool, sync::atomic::Ordering};
use std::{path::PathBuf, sync::Arc, sync::RwLock};

trait Serialize {
    fn to_buffer(&self, buffer: &mut Vec<u8>);
    fn make_message(&self) -> Vec<u8> {
        let mut buf = (0 as u32).to_le_bytes().to_vec();
        self.to_buffer(&mut buf);
        buf.splice(
            ..std::mem::size_of::<i32>(),
            (buf.len() as i32).to_le_bytes().iter().cloned(),
        );
        buf
    }
}

impl Serialize for bool {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        buffer.push(if *self { 1u8 } else { 0u8 });
    }
}

impl Serialize for i32 {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        buffer.extend(&self.to_le_bytes());
    }
}

impl Serialize for f32 {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        buffer.extend(&self.to_le_bytes());
    }
}

impl Serialize for CString {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        buffer.extend(self.to_bytes_with_nul());
    }
}

impl<T: Serialize> Serialize for Vec<T> {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        for elem in self {
            elem.to_buffer(buffer);
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone)]
enum TevControlHeader {
    OpenImage = 0,
    ReloadImage = 1,
    CloseImage = 2,
    UpdateImage = 3,
    CreateImage = 4,
}

impl Serialize for TevControlHeader {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        buffer.push(*self as u8);
    }
}

struct TevControlCreateImage {
    grab_focus: bool,
    image_name: CString,
    width: i32,
    height: i32,
    n_channels: i32,
    channel_names: Vec<CString>,
}

impl TevControlCreateImage {
    fn new_message(resolution: &na::Vector2<u32>, name: &str) -> Vec<u8> {
        TevControlCreateImage {
            grab_focus: true,
            image_name: CString::new(name).unwrap(),
            width: resolution.x as i32,
            height: resolution.y as i32,
            n_channels: 3,
            channel_names: vec![
                CString::new("r").unwrap(),
                CString::new("g").unwrap(),
                CString::new("b").unwrap(),
            ],
        }
        .make_message()
    }
}

impl Serialize for TevControlCreateImage {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        TevControlHeader::CreateImage.to_buffer(buffer);
        self.grab_focus.to_buffer(buffer);
        self.image_name.to_buffer(buffer);
        self.width.to_buffer(buffer);
        self.height.to_buffer(buffer);
        self.n_channels.to_buffer(buffer);
        self.channel_names.to_buffer(buffer);
    }
}

struct TevControlUpdateImage {
    grab_focus: bool,
    image_name: CString,
    channel: CString,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    image_data: Vec<f32>,
}

impl TevControlUpdateImage {
    fn new_message(film: &Film, name: &str) -> Vec<Vec<u8>> {
        const CHANNEL_NAMES: [&str; 3] = ["r", "g", "b"];
        let mut bufs = Vec::new();
        let channels = film.to_channel_updates();
        const CHUNK_DIM: usize = 100;
        for (idx, channel) in channels.iter().enumerate() {
            for (x, y) in (0..film.resolution.x as usize)
                .step_by(CHUNK_DIM)
                .cartesian_product((0..film.resolution.y as usize).step_by(CHUNK_DIM))
            {
                let cols = film.resolution.x as usize;
                let chunk_rows = CHUNK_DIM.min(film.resolution.y as usize - y);
                let chunk_cols = CHUNK_DIM.min(film.resolution.x as usize - x);
                let mut chunk_buf = Vec::new();
                for row in y..(y + chunk_rows) {
                    chunk_buf.extend_from_slice(
                        &channel[(row * cols + x)..(row * cols + x + chunk_cols)],
                    );
                }

                bufs.push(
                    TevControlUpdateImage {
                        grab_focus: true,
                        image_name: CString::new(name).unwrap(),
                        channel: CString::new(CHANNEL_NAMES[idx]).unwrap(),
                        x: x as i32,
                        y: y as i32,
                        width: chunk_cols as i32,
                        height: chunk_rows as i32,
                        image_data: chunk_buf,
                    }
                    .make_message(),
                );
            }
        }

        bufs
    }
}

impl Serialize for TevControlUpdateImage {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        TevControlHeader::UpdateImage.to_buffer(buffer);
        self.grab_focus.to_buffer(buffer);
        self.image_name.to_buffer(buffer);
        self.channel.to_buffer(buffer);
        self.x.to_buffer(buffer);
        self.y.to_buffer(buffer);
        self.width.to_buffer(buffer);
        self.height.to_buffer(buffer);
        self.image_data.to_buffer(buffer);
    }
}

enum ConnectionType {
    TCP(TcpStream),
    UDP(UdpSocket, String),
}

impl ConnectionType {
    fn write(&mut self, buf: &[u8]) -> anyhow::Result<()> {
        let bytes_sent = match self {
            ConnectionType::TCP(stream) => stream.write(buf)?,
            ConnectionType::UDP(socket, addr) => socket.send_to(buf, addr.as_str())?,
        };

        assert_eq!(bytes_sent, buf.len());

        Ok(())
    }
}

pub fn run(
    log: slog::Logger,
    render_scene: RenderScene,
    camera: Camera,
    integrator: PathIntegrator,
    output_path: PathBuf,
) -> anyhow::Result<()> {
    let server_address = "127.0.0.1:14158";
    let connection = if let Ok(stream) = TcpStream::connect("127.0.0.1:14158") {
        Some(ConnectionType::TCP(stream))
    } else if let Ok(socket) = UdpSocket::bind("0.0.0.0:0") {
        Some(ConnectionType::UDP(socket, String::from(server_address)))
    } else {
        None
    };

    if let Some(mut connection) = connection {
        let camera_master = Arc::new(RwLock::new(camera));
        let camera = camera_master.clone();
        let rendering_done_master = Arc::new(AtomicBool::new(false));
        let rendering_done = rendering_done_master.clone();
        let camera = camera.read().unwrap();

        connection
            .write(&TevControlCreateImage::new_message(&camera.film.resolution, "render")[..])?;

        let progressive_thread = std::thread::spawn(move || -> anyhow::Result<()> {
            let rendering_done = rendering_done_master;
            let camera = camera_master.read().unwrap();
            while !rendering_done.load(Ordering::Relaxed) {
                let buffers = TevControlUpdateImage::new_message(&camera.film, "render");
                for buf in buffers {
                    connection.write(&buf[..])?;
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }

            let buffers = TevControlUpdateImage::new_message(&camera.film, "render");
            for buf in buffers {
                connection.write(&buf[..])?;
            }

            Ok(())
        });

        integrator.render(&camera, &render_scene);
        rendering_done.store(true, Ordering::Relaxed);

        progressive_thread.join().unwrap()?;

        camera.film.to_rgba_image().save(&output_path).unwrap();
    } else {
        warn!(
            log,
            "could not conenct to display server, falling back to one shot rendering"
        );
        integrator.render(&camera, &render_scene);
        camera.film.to_rgba_image().save(&output_path).unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{convert::TryInto, ffi::CStr};

    use super::*;

    #[test]
    fn test_i32_to_buffer() {
        let mut buf = Vec::new();
        for i in -10000..10000 {
            i.to_buffer(&mut buf);
            let recovered = i32::from_le_bytes(buf[..4].try_into().unwrap());
            assert_eq!(i, recovered);
            buf.clear();
        }
    }

    #[test]
    fn test_tev_control_create_image() {
        let resolution = na::Vector2::new(1920, 1080);
        let image_name = "render";
        let message = TevControlCreateImage::new_message(&resolution, image_name);

        let mut curr_idx = 0;

        let mut next_idx = std::mem::size_of::<u32>();
        let payload_size = u32::from_le_bytes(message[curr_idx..next_idx].try_into().unwrap());
        assert_eq!(payload_size, message.len() as u32);
        curr_idx = next_idx;

        next_idx += std::mem::size_of::<u8>();
        let control_type = u8::from_le_bytes(message[curr_idx..next_idx].try_into().unwrap());
        assert_eq!(TevControlHeader::CreateImage as u8, control_type);
        curr_idx = next_idx;

        next_idx += image_name.len() + 1;
        let name = CStr::from_bytes_with_nul(&message[curr_idx..next_idx]).unwrap();
        assert_eq!(name.to_str().unwrap(), image_name);
        curr_idx = next_idx;

        next_idx += std::mem::size_of::<i8>();
        let focus = i8::from_le_bytes(message[curr_idx..next_idx].try_into().unwrap());
        assert_eq!(focus, 1);
        curr_idx = next_idx;

        next_idx += std::mem::size_of::<i32>();
        let x_res = i32::from_le_bytes(message[curr_idx..next_idx].try_into().unwrap());
        assert_eq!(x_res as u32, resolution.x);
        curr_idx = next_idx;

        next_idx += std::mem::size_of::<i32>();
        let y_res = i32::from_le_bytes(message[curr_idx..next_idx].try_into().unwrap());
        assert_eq!(y_res as u32, resolution.y);
        curr_idx = next_idx;
    }
}
