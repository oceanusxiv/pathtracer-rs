lazy_static::lazy_static! {
    pub static ref VERTEX: String =
    "
#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;

layout(binding=0)
uniform Uniforms {
    mat4 u_view_proj;
};

layout(location=0) out vec3 v_position;
layout(location=1) out vec3 v_normal;

void main() {
    v_normal = a_normal;
    v_position = a_position;
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
    ".to_string();

    pub static ref FRAGMENT: String =
    "
#version 450

layout(location=0) in vec3 v_position;
layout(location=1) in vec3 v_normal;

layout(location=0) out vec4 f_color;

void main() {
    vec4 object_color = vec4(0.0, 1.0, 1.0, 1.0);
    vec3 light_color = vec3(1.0, 1.0, 1.0);
    vec3 light_position = vec3(10.0, -10.0, 10.0);

    float ambient_strength = 0.1;
    vec3 ambient_color = light_color * ambient_strength;

    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(light_position - v_position);

    float diffuse_strength = max(dot(normal, light_dir), 0.0);
    vec3 diffuse_color = light_color * diffuse_strength;

    vec3 result = (ambient_color + diffuse_color) * object_color.xyz;

    f_color = vec4(result, object_color.a);
}
    ".to_string();
}