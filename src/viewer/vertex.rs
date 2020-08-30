pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VertexPos {
    position: glm::Vec3,
}

unsafe impl bytemuck::Zeroable for VertexPos {}

unsafe impl bytemuck::Pod for VertexPos {}

impl From<&na::Point3<f32>> for VertexPos {
    fn from(position: &na::Point3<f32>) -> Self {
        VertexPos {
            position: position.coords,
        }
    }
}

impl Vertex for VertexPos {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<VertexPos>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[wgpu::VertexAttributeDescriptor {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float3,
            }],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VertexPosTex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

unsafe impl bytemuck::Zeroable for VertexPosTex {}

unsafe impl bytemuck::Pod for VertexPosTex {}

impl Vertex for VertexPosTex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<VertexPosTex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<glm::Vec3>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VertexPosNorm {
    position: glm::Vec3,
    normal: glm::Vec3,
}

unsafe impl bytemuck::Zeroable for VertexPosNorm {}

unsafe impl bytemuck::Pod for VertexPosNorm {}

impl From<(&na::Point3<f32>, &na::Vector3<f32>)> for VertexPosNorm {
    fn from(pair: (&na::Point3<f32>, &na::Vector3<f32>)) -> Self {
        let (position, normal) = pair;
        VertexPosNorm {
            position: position.coords,
            normal: *normal,
        }
    }
}

impl Vertex for VertexPosNorm {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<VertexPosNorm>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<glm::Vec3>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float3,
                },
            ],
        }
    }
}
