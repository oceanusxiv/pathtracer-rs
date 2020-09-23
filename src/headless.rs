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
    image_name: CString,
    grab_focus: bool,
    width: i32,
    height: i32,
    n_channels: i32,
    channel_names: Vec<CString>,
}

impl TevControlCreateImage {
    fn buffer_from_film(film: &Film, name: &str) -> Vec<u8> {
        let mut buffer = vec![];
        TevControlCreateImage {
            image_name: CString::new(name).unwrap(),
            grab_focus: true,
            width: film.resolution.x as i32,
            height: film.resolution.x as i32,
            n_channels: 3,
            channel_names: vec![
                CString::new("r").unwrap(),
                CString::new("g").unwrap(),
                CString::new("b").unwrap(),
            ],
        }
        .to_buffer(&mut buffer);

        buffer
    }
}

impl Serialize for TevControlCreateImage {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        TevControlHeader::CreateImage.to_buffer(buffer);
        self.image_name.to_buffer(buffer);
        self.grab_focus.to_buffer(buffer);
        self.width.to_buffer(buffer);
        self.height.to_buffer(buffer);
        self.n_channels.to_buffer(buffer);
        self.channel_names.to_buffer(buffer);
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
    fn buffers_from_film(film: &Film, name: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (r, g, b) = film.to_channel_updates();
        let mut r_buf = vec![];
        TevControlUpdateImage {
            image_name: CString::new(name).unwrap(),
            grab_focus: true,
            channel: CString::new("r").unwrap(),
            x: 0,
            y: 0,
            width: film.resolution.x as i32,
            height: film.resolution.x as i32,
            image_data: r,
        }
        .to_buffer(&mut r_buf);

        let mut g_buf = vec![];
        TevControlUpdateImage {
            image_name: CString::new(name).unwrap(),
            grab_focus: true,
            channel: CString::new("g").unwrap(),
            x: 0,
            y: 0,
            width: film.resolution.x as i32,
            height: film.resolution.x as i32,
            image_data: g,
        }
        .to_buffer(&mut g_buf);

        let mut b_buf = vec![];
        TevControlUpdateImage {
            image_name: CString::new(name).unwrap(),
            grab_focus: true,
            channel: CString::new("b").unwrap(),
            x: 0,
            y: 0,
            width: film.resolution.x as i32,
            height: film.resolution.x as i32,
            image_data: b,
        }
        .to_buffer(&mut b_buf);

        (r_buf, g_buf, b_buf)
    }
}

impl Serialize for TevControlUpdateImage {
    fn to_buffer(&self, buffer: &mut Vec<u8>) {
        TevControlHeader::UpdateImage.to_buffer(buffer);
        self.image_name.to_buffer(buffer);
        self.grab_focus.to_buffer(buffer);
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
        match self {
            ConnectionType::TCP(stream) => {
                stream.write(buf)?;
            }
            ConnectionType::UDP(socket, addr) => {
                socket.send_to(buf, addr.as_str())?;
            }
        }

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

        connection.write(&TevControlCreateImage::buffer_from_film(&camera.film, "render")[..])?;

        let progressive_thread = std::thread::spawn(move || -> anyhow::Result<()> {
            let rendering_done = rendering_done_master;
            let camera = camera_master.read().unwrap();
            while !rendering_done.load(Ordering::Relaxed) {
                let (r_cmd, g_cmd, b_cmd) =
                    TevControlUpdateImage::buffers_from_film(&camera.film, "render");
                connection.write(&r_cmd[..])?;
                connection.write(&g_cmd[..])?;
                connection.write(&b_cmd[..])?;
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
