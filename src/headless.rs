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
    fn to_u8_vec(&self) -> Vec<u8>;
    fn make_message(&self) -> Vec<u8> {
        let mut payload = self.to_u8_vec();
        let mut buf = (payload.len() as u32).to_le_bytes().to_vec();
        buf.append(&mut payload);
        buf
    }
}

impl Serialize for bool {
    fn to_u8_vec(&self) -> Vec<u8> {
        if *self {
            vec![1u8]
        } else {
            vec![0u8]
        }
    }
}

impl Serialize for i32 {
    fn to_u8_vec(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Serialize for f32 {
    fn to_u8_vec(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Serialize for CString {
    fn to_u8_vec(&self) -> Vec<u8> {
        self.to_bytes_with_nul().to_vec()
    }
}

impl<T: Serialize> Serialize for Vec<T> {
    fn to_u8_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for elem in self {
            buf.append(&mut elem.to_u8_vec());
        }

        buf
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
    fn to_u8_vec(&self) -> Vec<u8> {
        vec![*self as u8]
    }
}

struct TevControlCreateImage {
    image_name: CString,
    grab_focus: bool,
    width: i32,
    height: i32,
    n_channels: i32,
    channel_names: Vec<CString>,
}

impl TevControlCreateImage {
    fn new_message(resolution: &na::Vector2<u32>, name: &str) -> Vec<u8> {
        TevControlCreateImage {
            image_name: CString::new(name).unwrap(),
            grab_focus: true,
            width: resolution.x as i32,
            height: resolution.x as i32,
            n_channels: 3,
            channel_names: vec![
                CString::new("0").unwrap(),
                CString::new("1").unwrap(),
                CString::new("2").unwrap(),
            ],
        }
        .make_message()
    }
}

impl Serialize for TevControlCreateImage {
    fn to_u8_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.append(&mut TevControlHeader::CreateImage.to_u8_vec());
        buf.append(&mut self.image_name.to_u8_vec());
        buf.append(&mut self.grab_focus.to_u8_vec());
        buf.append(&mut self.width.to_u8_vec());
        buf.append(&mut self.height.to_u8_vec());
        buf.append(&mut self.n_channels.to_u8_vec());
        buf.append(&mut self.channel_names.to_u8_vec());

        buf
    }
}

struct TevControlUpdateImage {
    image_name: CString,
    grab_focus: bool,
    channel: CString,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    image_data: Vec<f32>,
}

impl TevControlUpdateImage {
    fn buffers_from_film(film: &Film, name: &str) -> Vec<Vec<u8>> {
        let mut bufs = Vec::new();
        let (r, g, b) = film.to_channel_updates();
        const CHUNK_DIM: usize = 10;
        for (idx, channel) in [r, g, b].iter().enumerate() {
            for (x, y) in (0..film.resolution.x as usize)
                .step_by(CHUNK_DIM)
                .cartesian_product((0..film.resolution.y as usize).step_by(CHUNK_DIM))
            {
                let mut chunk_buf = Vec::new();

                for i in 0..CHUNK_DIM.min(film.resolution.y as usize - y) {
                    chunk_buf.extend_from_slice(
                        &channel[(i * CHUNK_DIM + x)
                            ..(i * CHUNK_DIM + x + CHUNK_DIM.min(film.resolution.x as usize - x))],
                    );
                }

                bufs.push(
                    TevControlUpdateImage {
                        image_name: CString::new(name).unwrap(),
                        grab_focus: true,
                        channel: CString::new(idx.to_string()).unwrap(),
                        x: x as i32,
                        y: y as i32,
                        width: CHUNK_DIM as i32,
                        height: CHUNK_DIM as i32,
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
    fn to_u8_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.append(&mut TevControlHeader::UpdateImage.to_u8_vec());
        buf.append(&mut self.image_name.to_u8_vec());
        buf.append(&mut self.grab_focus.to_u8_vec());
        buf.append(&mut self.channel.to_u8_vec());
        buf.append(&mut self.x.to_u8_vec());
        buf.append(&mut self.y.to_u8_vec());
        buf.append(&mut self.width.to_u8_vec());
        buf.append(&mut self.height.to_u8_vec());
        buf.append(&mut self.image_data.to_u8_vec());

        buf
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
                // let buffers = TevControlUpdateImage::buffers_from_film(&camera.film, "render");
                // for buf in buffers {
                //     connection.write(&buf[..])?;
                // }
                std::thread::sleep(std::time::Duration::from_secs(2));
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
    use std::convert::TryInto;

    use super::*;

    #[test]
    fn test_tev_control_create_image() {
        let resolution = na::Vector2::new(1920, 1080);
        let message = TevControlCreateImage::new_message(&resolution, "render");

        let payload_size = u32::from_le_bytes(message[..4].try_into().unwrap());
        assert_eq!(payload_size, (message.len() - 4) as u32);
    }
}
