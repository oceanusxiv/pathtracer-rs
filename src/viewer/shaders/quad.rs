lazy_static::lazy_static! {
    static ref VERTEX: String =
    "
#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_tex_coords;

layout(location=0) out vec2 v_tex_coords;

void main() {
    v_tex_coords = a_tex_coords;
    gl_Position = vec4(a_position, 1.0);
}
    ".to_string();

    static ref FRAGMENT: String =
    "
#version 450

layout(location=0) in vec2 v_tex_coords;
layout(location=1) in vec3 v_color;

layout(location=0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

void main() {
    f_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);
}
    ".to_string();
}

pub fn compile_shaders(
    compiler: &mut shaderc::Compiler,
    device: &wgpu::Device,
) -> (wgpu::ShaderModule, wgpu::ShaderModule) {
    let vert = super::compile_shader(
        &VERTEX,
        "quad.vert",
        shaderc::ShaderKind::Vertex,
        compiler,
        device,
    );
    let frag = super::compile_shader(
        &FRAGMENT,
        "quad.frag",
        shaderc::ShaderKind::Fragment,
        compiler,
        device,
    );
    (vert, frag)
}
