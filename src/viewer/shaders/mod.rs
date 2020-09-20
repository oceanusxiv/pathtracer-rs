pub mod flat;
pub mod flat_instance;
pub mod phong;
pub mod quad;

fn compile_shader(
    source_text: &str,
    tag: &str,
    shader_kind: shaderc::ShaderKind,
    compiler: &mut shaderc::Compiler,
    device: &wgpu::Device,
) -> wgpu::ShaderModule {
    let vs_spirv = compiler
        .compile_into_spirv(source_text, shader_kind, tag, "main", None)
        .unwrap();
    let vs_data = wgpu::util::make_spirv(vs_spirv.as_binary_u8());
    device.create_shader_module(vs_data)
}
