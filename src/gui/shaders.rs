use rendy::shader::{
    ShaderKind, ShaderSet, ShaderSetBuilder, SourceLanguage, SourceShaderInfo, SpirvShader,
};

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SourceShaderInfo::new(
    "
    #version 450
    #extension GL_ARB_separate_shader_objects : enable
    layout(location = 0) in vec3 pos;
    layout(location = 1) in vec4 color;
    layout(location = 0) out vec4 frag_color;
    void main() {
        frag_color = color;
        gl_Position = vec4(pos, 1.0);
    }
    ",
        "triangle.vert",
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = SourceShaderInfo::new(
    "
    #version 450
    #extension GL_ARB_separate_shader_objects : enable
    layout(early_fragment_tests) in;
    layout(location = 0) in vec4 frag_color;
    layout(location = 0) out vec4 color;
    void main() {
        color = frag_color;
    }
    ",
        "triangle.frag",
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    pub static ref SHADERS: ShaderSetBuilder = ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

#[cfg(feature = "spirv-reflection")]
use rendy::shader::SpirvReflection;

#[cfg(feature = "spirv-reflection")]
lazy_static::lazy_static! {
    pub static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}
