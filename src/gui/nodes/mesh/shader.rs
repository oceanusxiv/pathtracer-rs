use rendy::shader::{
    ShaderKind, ShaderSet, ShaderSetBuilder, SourceLanguage, SourceShaderInfo, SpirvShader,
};

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SourceShaderInfo::new(
    "
    #version 450
    #extension GL_ARB_separate_shader_objects : enable
    
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec4 color;
    layout(location = 2) in vec3 normal;
    // vec4[4] is used instead of mat4 due to spirv-cross bug for dx12 backend
    layout(location = 3) in vec4 model[4]; // per-instance.
    
    layout(set = 0, binding = 0) uniform Args {
        mat4 proj;
        mat4 view;
    };
    
    layout(location = 0) out vec4 frag_pos;
    layout(location = 1) out vec3 frag_norm;
    layout(location = 2) out vec4 frag_color;
    
    void main() {
        mat4 model_mat = mat4(model[0], model[1], model[2], model[3]);
        frag_color = color;
        frag_norm = normalize((vec4(normal, 1.0) * model_mat).xyz);
        frag_pos = model_mat * vec4(position, 1.0);
        gl_Position = proj * view * frag_pos;
    }
    ",
        "mesh.vert",
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = SourceShaderInfo::new(
    "
    #version 450
    #extension GL_ARB_separate_shader_objects : enable
    
    layout(early_fragment_tests) in;
    
    layout(location = 0) in vec4 in_pos;
    layout(location = 1) in vec3 frag_norm;
    layout(location = 2) in vec4 frag_color;
    layout(location = 0) out vec4 color;
    
    layout(std140, set = 0, binding = 0) uniform Args {
        mat4 proj;
        mat4 view;
    };
    
    void main() {    
        vec3 frag_pos = in_pos.xyz / in_pos.w;
        color = frag_color;
    }
    ",
        "mesh.frag",
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    pub static ref SHADERS: ShaderSetBuilder = ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

use rendy::shader::SpirvReflection;

lazy_static::lazy_static! {
    pub static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}
