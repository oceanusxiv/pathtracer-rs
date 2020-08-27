pub mod flat;
pub mod phong;

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
    let vs_data = wgpu::read_spirv(std::io::Cursor::new(vs_spirv.as_binary_u8())).unwrap();
    device.create_shader_module(&vs_data)
}
