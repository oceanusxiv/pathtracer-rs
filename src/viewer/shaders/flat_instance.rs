lazy_static::lazy_static! {
    static ref VERTEX: String =
    "
#version 450

layout(location=0) in vec3 a_position;

layout(binding=0)
uniform Uniforms {
    mat4 u_view_proj;
};
layout(set=1, binding=0)
buffer Instances {
    mat4 s_models[];
};


void main() {
    gl_Position = u_view_proj * s_models[gl_InstanceIndex] * vec4(a_position, 1.0);
}
    ".to_string();

    static ref FRAGMENT: String =
    "
#version 450

layout(location=0) out vec4 f_color;

void main() {
    f_color = vec4(0.0, 0.0, 0.0, 1.0);
}
    ".to_string();
}

pub fn compile_shaders(
    compiler: &mut shaderc::Compiler,
    device: &wgpu::Device,
) -> (wgpu::ShaderModule, wgpu::ShaderModule) {
    let vert = super::compile_shader(
        &VERTEX,
        "bounds.vert",
        shaderc::ShaderKind::Vertex,
        compiler,
        device,
    );
    let frag = super::compile_shader(
        &FRAGMENT,
        "bounds.frag",
        shaderc::ShaderKind::Fragment,
        compiler,
        device,
    );
    (vert, frag)
}
