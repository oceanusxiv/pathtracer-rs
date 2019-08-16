#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]

mod nodes;

use rendy::{
    command::{Families, Graphics, QueueId, RenderPassEncoder, Supports},
    factory::{Config, Factory, ImageState},
    graph::{
        present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    hal,
    memory::Dynamic,
    mesh::PosColorNorm,
    resource::{Buffer, BufferInfo, DescriptorSetLayout, Escape, Handle},
    shader::{
        ShaderKind, ShaderSet, ShaderSetBuilder, SourceLanguage, SourceShaderInfo, SpirvShader,
    },
    wsi::winit::{self, Event, EventsLoop, WindowBuilder, WindowEvent},
};

use genmesh::generators::{IndexedPolygon, SharedVertex};
use nodes::mesh::pipeline::MeshRenderPipeline;
use std::{collections::HashSet, time};

use crate::common::*;

#[cfg(feature = "dx12")]
pub type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
pub type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
pub type Backend = rendy::vulkan::Backend;

#[cfg(feature = "empty")]
pub type Backend = rendy::empty::Backend;

struct RenderWorld<B: hal::Backend> {
    pub world: World,
    pub meshes: Vec<rendy::mesh::Mesh<B>>,
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main_loop<W>(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    mut graph: Graph<Backend, W>,
    world: &mut W,
) -> Result<(), failure::Error> {
    let start_time = time::Instant::now();
    let mut checkpoint = start_time;
    let mut frames = 0u64..;
    let mut should_close = false;

    #[cfg(feature = "rd")]
    rd.start_frame_capture(std::ptr::null(), std::ptr::null());

    while !should_close {
        let start = frames.start;

        for _ in &mut frames {
            factory.maintain(families);
            event_loop.poll_events(|event| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => should_close = true,
                _ => (),
            });
            graph.run(factory, families, &world);

            let elapsed = checkpoint.elapsed();

            if should_close || elapsed > std::time::Duration::new(2, 0) {
                let frames = frames.start - start;
                let nanos = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;
                log::info!("FPS: {}", frames * 1_000_000_000 / nanos);
                checkpoint += elapsed;
                break;
            }
        }
    }

    graph.dispose(factory, &world);
    Ok(())
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
pub fn run_gui() -> Result<(), failure::Error> {
    println!("Hello, world!");

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config)?;

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("pathtracer-rs")
        .build(&event_loop)?;

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, RenderWorld<Backend>>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());
    let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);
    let aspect = size.width / size.height;

    let color = graph_builder.create_image(
        window_kind,
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue::Color([1.0, 1.0, 1.0, 1.0].into())),
    );

    let depth = graph_builder.create_image(
        window_kind,
        1,
        hal::format::Format::D16Unorm,
        Some(hal::command::ClearValue::DepthStencil(
            hal::command::ClearDepthStencil(1.0, 0),
        )),
    );

    let pass = graph_builder.add_node(
        MeshRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .with_depth_stencil(depth)
            .into_pass(),
    );

    let present_builder = PresentNode::builder(&factory, surface, color).with_dependency(pass);

    let frames = present_builder.image_count();

    graph_builder.add_node(present_builder);

    let mut render_world = RenderWorld {
        world: World {
            camera: Camera {
                projection: nalgebra::Perspective3::new(aspect as f32, 3.1415 / 4.0, 1.0, 200.0),
                view: nalgebra::Projective3::identity()
                    * nalgebra::Translation3::new(0.0, 0.0, 10.0),
            },
            object_transforms: vec![
                nalgebra::Transform3::identity() * nalgebra::Translation3::new(0.0, 0.0, -10.0),
            ],
        },
        meshes: vec![],
    };

    let icosphere = genmesh::generators::IcoSphere::subdivide(4);
    let indices: Vec<_> = genmesh::Vertices::vertices(icosphere.indexed_polygon_iter())
        .map(|i| i as u32)
        .collect();
    let vertices: Vec<_> = icosphere
        .shared_vertex_iter()
        .map(|v| PosColorNorm {
            position: v.pos.into(),
            color: [
                (v.pos.x + 1.0) / 2.0,
                (v.pos.y + 1.0) / 2.0,
                (v.pos.z + 1.0) / 2.0,
                1.0,
            ]
            .into(),
            normal: v.normal.into(),
        })
        .collect();

    let graph = graph_builder.with_frames_in_flight(frames).build(
        &mut factory,
        &mut families,
        &mut render_world,
    )?;

    render_world.meshes.push(
        rendy::mesh::Mesh::<Backend>::builder()
            .with_indices(&indices[..])
            .with_vertices(&vertices[..])
            .build(graph.node_queue(pass), &factory)
            .unwrap(),
    );

    main_loop(
        &mut event_loop,
        &mut factory,
        &mut families,
        graph,
        &mut render_world,
    )
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
pub fn run_gui() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
