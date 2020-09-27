use crate::pathtracer::RenderScene;
use anyhow::Context;
use optix::DeviceStorage;
use ustr::ustr;

fn init_optix() -> Result<(), Box<dyn std::error::Error>> {
    cu::init()?;
    let device_count = cu::Device::get_count()?;
    if device_count == 0 {
        panic!("No CUDA devices found!");
    }

    optix::init()?;
    Ok(())
}

#[repr(C)]
#[derive(Copy, Clone)]
struct LaunchParams {
    pub frame_id: i32,
    pub traversable: optix::TraversableHandle,
}

unsafe impl optix::DeviceCopy for LaunchParams {}

type RaygenRecord = optix::SbtRecord<i32>;
type MissRecord = optix::SbtRecord<i32>;
struct HitgroupSbtData {
    object_id: u32,
}
unsafe impl optix::DeviceCopy for HitgroupSbtData {}
type HitgroupRecord = optix::SbtRecord<HitgroupSbtData>;

pub struct OptixAccelerator {
    cuda_context: cu::Context,
    optix_context: optix::DeviceContext,
    stream: cu::Stream,
    launch_params: optix::DeviceVariable<LaunchParams>,
    buf_raygen: optix::TypedBuffer<RaygenRecord>,
    buf_hitgroup: optix::TypedBuffer<HitgroupRecord>,
    buf_miss: optix::TypedBuffer<MissRecord>,
    as_handle: optix::TraversableHandle,
    as_buffer: optix::Buffer,
    sbt: optix::sys::OptixShaderBindingTable,
    pipeline: optix::Pipeline,
}

fn build_optix_as(
    scene: &RenderScene,
    ctx: &optix::DeviceContext,
    stream: &cu::Stream,
) -> anyhow::Result<(optix::TraversableHandle, optix::Buffer)> {
    // create geometry and accels
    let buf_vertex: Vec<optix::TypedBuffer<na::Point3<f32>, cu::DefaultDeviceAlloc>> = scene
        .meshes
        .iter()
        .map(|m| optix::TypedBuffer::from_slice(&m.pos))
        .collect::<Result<Vec<_>, optix::Error>>()
        .context("allocating vertex buffer")?;

    let buf_index: Vec<optix::TypedBuffer<na::Vector3<u32>, cu::DefaultDeviceAlloc>> = scene
        .meshes
        .iter()
        .map(|m| optix::TypedBuffer::from_slice(&m.indices))
        .collect::<Result<Vec<_>, optix::Error>>()
        .context("Allocating index buffer")?;

    let geometry_flags = optix::GeometryFlags::None;
    let triangle_inputs: Vec<_> = buf_vertex
        .iter()
        .zip(&buf_index)
        .map(
            |(vertex, index): (
                &optix::TypedBuffer<na::Point3<f32>>,
                &optix::TypedBuffer<na::Vector3<u32>>,
            )| {
                optix::BuildInput::TriangleArray(
                    optix::TriangleArray::new(
                        std::slice::from_ref(vertex),
                        std::slice::from_ref(&geometry_flags),
                    )
                    .index_buffer(index),
                )
            },
        )
        .collect();

    // blas setup
    let accel_options = optix::AccelBuildOptions::new(
        optix::BuildFlags::ALLOW_COMPACTION,
        optix::BuildOperation::Build,
    );

    let blas_buffer_sizes = ctx
        .accel_compute_memory_usage(&[accel_options], &triangle_inputs)
        .context("Accel compute memory usage")?;

    // prepare compaction
    // we need scratch space for the BVH build which we allocate here as
    // an untyped buffer. Note that we need to specify the alignment
    let temp_buffer = optix::Buffer::uninitialized_with_align_in(
        blas_buffer_sizes.temp_size_in_bytes,
        optix::ACCEL_BUFFER_BYTE_ALIGNMENT,
        cu::DefaultDeviceAlloc,
    )?;

    let output_buffer = optix::Buffer::uninitialized_with_align_in(
        blas_buffer_sizes.output_size_in_bytes,
        optix::ACCEL_BUFFER_BYTE_ALIGNMENT,
        cu::DefaultDeviceAlloc,
    )?;

    // DeviceVariable is a type that wraps a POD type to allow easy access
    // to the data rather than having to carry around the host type and a
    // separate device allocation for it
    let mut compacted_size = optix::DeviceVariable::new(0usize)?;

    // tell the accel build we want to know the size the compacted buffer
    // will be
    let mut properties = vec![optix::AccelEmitDesc::CompactedSize(
        compacted_size.device_ptr(),
    )];

    // build the bvh
    let as_handle = ctx
        .accel_build(
            &stream,
            &[accel_options],
            &triangle_inputs,
            &temp_buffer,
            &output_buffer,
            &mut properties,
        )
        .context("accel build")?;

    cu::Context::synchronize().context("Accel build sync")?;

    // copy the size back from the device, we can now treat it as
    // the underlying type by `Deref`
    compacted_size.download()?;

    // allocate the final acceleration structure storage
    let as_buffer = optix::Buffer::uninitialized_with_align_in(
        *compacted_size,
        optix::ACCEL_BUFFER_BYTE_ALIGNMENT,
        cu::DefaultDeviceAlloc,
    )?;

    // compact the accel.
    // we don't need the original handle any more
    let as_handle = ctx
        .accel_compact(&stream, as_handle, &as_buffer)
        .context("Accel compact")?;
    cu::Context::synchronize().context("Accel compact sync")?;

    Ok((as_handle, as_buffer))
}

impl OptixAccelerator {
    pub fn new(scene: &RenderScene) -> Result<Self, Box<dyn std::error::Error>> {
        init_optix()?;

        // create CUDA and OptiX contexts
        let device = cu::Device::get(0)?;
        let tex_align = device.get_attribute(cu::DeviceAttribute::TextureAlignment)?;
        let srf_align = device.get_attribute(cu::DeviceAttribute::SurfaceAlignment)?;
        println!("tex align: {}\nsrf align: {}", tex_align, srf_align);

        let cuda_context =
            device.ctx_create(cu::ContextFlags::SCHED_AUTO | cu::ContextFlags::MAP_HOST)?;
        let stream = cu::Stream::create(cu::StreamFlags::DEFAULT)?;

        let mut ctx = optix::DeviceContext::create(&cuda_context)?;
        ctx.set_log_callback(|_level, tag, msg| println!("[{}]: {}", tag, msg), 4);

        // create module
        let module_compile_options = optix::ModuleCompileOptions {
            max_register_count: 50,
            opt_level: optix::CompileOptimizationLevel::Default,
            debug_level: optix::CompileDebugLevel::None,
        };

        let pipeline_compile_options = optix::PipelineCompileOptions::new()
            .uses_motion_blur(false)
            .num_attribute_values(2)
            .num_payload_values(2)
            .traversable_graph_flags(optix::TraversableGraphFlags::ALLOW_SINGLE_GAS)
            .exception_flags(optix::ExceptionFlags::NONE)
            .pipeline_launch_params_variable_name(ustr("optixLaunchParams"));

        let ptx = include_str!(concat!(
            env!("OUT_DIR"),
            "/src/pathtracer/accelerator/device_programs.ptx"
        ));

        let (module, _log) =
            ctx.module_create_from_ptx(&module_compile_options, &pipeline_compile_options, ptx)?;

        // create raygen program
        let pgdesc_raygen = optix::ProgramGroupDesc::raygen(&module, ustr("__raygen__renderFrame"));

        let (pg_raygen, _log) = ctx.program_group_create(&[pgdesc_raygen])?;

        // create miss program
        let pgdesc_miss = optix::ProgramGroupDesc::miss(&module, ustr("__miss__radiance"));

        let (pg_miss, _log) = ctx.program_group_create(&[pgdesc_miss])?;

        let pgdesc_hitgroup = optix::ProgramGroupDesc::hitgroup(
            Some((&module, ustr("__closesthit__radiance"))),
            Some((&module, ustr("__anyhit__radiance"))),
            None,
        );

        // create hitgroup programs
        let (pg_hitgroup, _log) = ctx.program_group_create(&[pgdesc_hitgroup])?;

        let (as_handle, as_buffer) = build_optix_as(scene, &ctx, &stream)?;

        // create pipeline
        let mut program_groups = Vec::new();
        program_groups.extend(pg_raygen.iter().cloned());
        program_groups.extend(pg_miss.iter().cloned());
        program_groups.extend(pg_hitgroup.iter().cloned());

        let pipeline_link_options = optix::PipelineLinkOptions {
            max_trace_depth: 2,
            debug_level: optix::CompileDebugLevel::LineInfo,
        };

        let (pipeline, _log) = ctx.pipeline_create(
            &pipeline_compile_options,
            pipeline_link_options,
            &program_groups,
        )?;

        pipeline.set_stack_size(2 * 1024, 2 * 1024, 2 * 1024, 1)?;

        // create SBT
        let rec_raygen: Vec<_> = pg_raygen
            .iter()
            .map(|pg| RaygenRecord::pack(0, pg).expect("failed to pack raygen record"))
            .collect();

        let rec_miss: Vec<_> = pg_miss
            .iter()
            .map(|pg| MissRecord::pack(0, pg).expect("failed to pack miss record"))
            .collect();

        let num_objects = 1;
        let rec_hitgroup: Vec<_> = (0..num_objects)
            .map(|i| {
                let object_type = 0;
                let rec = HitgroupRecord::pack(
                    HitgroupSbtData { object_id: i },
                    &pg_hitgroup[object_type],
                )
                .expect("failed to pack hitgroup record");
                rec
            })
            .collect();

        let buf_raygen = optix::TypedBuffer::from_slice(&rec_raygen)?;
        let buf_miss = optix::TypedBuffer::from_slice(&rec_miss)?;
        let buf_hitgroup = optix::TypedBuffer::from_slice(&rec_hitgroup)?;

        let sbt = optix::ShaderBindingTable::new(&buf_raygen)
            .miss(&buf_miss)
            .hitgroup(&buf_hitgroup)
            .build();

        let launch_params = optix::DeviceVariable::new(LaunchParams {
            frame_id: 0,
            traversable: as_handle,
        })?;

        Ok(Self {
            cuda_context,
            optix_context: ctx,
            stream,
            launch_params,
            buf_raygen,
            buf_hitgroup,
            buf_miss,
            as_handle,
            as_buffer,
            sbt,
            pipeline,
        })
    }

    pub fn intersect(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.launch_params.upload()?;
        self.launch_params.frame_id += 1;

        optix::launch(
            &self.pipeline,
            &self.stream,
            &self.launch_params,
            &self.sbt,
            1,
            1,
            1,
        )?;

        cu::Context::synchronize()?;

        Ok(())
    }
}
