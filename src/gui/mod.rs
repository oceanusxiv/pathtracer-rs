#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]

mod pipelines;
mod shaders;

use rendy::{
    command::{Families, Graphics, QueueId, RenderPassEncoder, Supports},
    factory::{Config, Factory, ImageState},
    graph::{
        present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    hal,
    memory::Dynamic,
    mesh::PosColor,
    resource::{Buffer, BufferInfo, DescriptorSetLayout, Escape, Handle},
    shader::{
        ShaderKind, ShaderSet, ShaderSetBuilder, SourceLanguage, SourceShaderInfo, SpirvShader,
    },
    wsi::winit::{self, Event, EventsLoop, WindowBuilder, WindowEvent},
};

use pipelines::TriangleRenderPipeline;

use std::{collections::HashSet, time};

#[cfg(feature = "dx12")]
pub type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
pub type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
pub type Backend = rendy::vulkan::Backend;

#[cfg(feature = "empty")]
pub type Backend = rendy::empty::Backend;

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main_loop(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    mut graph: Graph<Backend, ()>,
) -> Result<(), failure::Error> {
    let start_time = time::Instant::now();
    let mut checkpoint = start_time;
    let mut frames = 0u64..;
    let mut should_close = false;

    while !should_close {
        let start = frames.start;

        for _ in &mut frames {
            factory.maintain(families);
            event_loop.poll_events(|_| ());
            graph.run(factory, families, &mut ());

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

    graph.dispose(factory, &mut ());
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

    let mut graph_builder = GraphBuilder::<Backend, ()>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let color = graph_builder.create_image(
        hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1),
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue::Color([1.0, 1.0, 1.0, 1.0].into())),
    );

    let pass = graph_builder.add_node(
        TriangleRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    graph_builder.add_node(PresentNode::builder(&factory, surface, color).with_dependency(pass));

    let graph = graph_builder.build(&mut factory, &mut families, &mut ())?;

    main_loop(&mut event_loop, &mut factory, &mut families, graph)
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
pub fn run_gui() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
