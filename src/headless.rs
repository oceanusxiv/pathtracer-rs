use crate::{
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

#[repr(C)]
struct TevControlCreateImage {
    image_name: CString,
    grab_focus: bool,
    width: i32,
    height: i32,
    n_channels: i32,
    channel_names: Vec<CString>,
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

pub fn run(
    log: slog::Logger,
    render_scene: RenderScene,
    camera: Camera,
    integrator: PathIntegrator,
    output_path: PathBuf,
) -> anyhow::Result<()> {
    if let Ok(stream) = TcpStream::connect("127.0.0.1:14158") {
        panic!("tcp not supported yet");
    } else if let Ok(stream) = UdpSocket::bind("0.0.0.0:0") {
        let camera_master = Arc::new(RwLock::new(camera));
        let camera = camera_master.clone();
        let rendering_done_master = Arc::new(AtomicBool::new(false));
        let rendering_done = rendering_done_master.clone();
        let camera = camera.read().unwrap();

        let mut buffer = vec![];
        TevControlCreateImage {
            image_name: CString::new("render").unwrap(),
            grab_focus: true,
            width: camera.film.resolution.x as i32,
            height: camera.film.resolution.x as i32,
            n_channels: 3,
            channel_names: vec![
                CString::new("r").unwrap(),
                CString::new("g").unwrap(),
                CString::new("b").unwrap(),
            ],
        }
        .to_buffer(&mut buffer);

        stream.send_to(&buffer[..], "127.0.0.1:14158")?;

        std::thread::spawn(move || {
            let rendering_done = rendering_done_master;
            let camera = camera_master.read().unwrap();
            while !rendering_done.load(Ordering::Relaxed) {
                let image = camera.film.write_image();

                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        });

        integrator.render(&camera, &render_scene);
        rendering_done.store(true, Ordering::Relaxed);
        camera.film.write_image().save(&output_path).unwrap();
    } else {
        warn!(
            log,
            "could not conenct to display server, falling back to one shot rendering"
        );
        integrator.render(&camera, &render_scene);
        camera.film.write_image().save(&output_path).unwrap();
    }

    Ok(())
}
