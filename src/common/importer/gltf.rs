use crate::common::{bounds::Bounds3, Camera, DEFAULT_Z_FAR, DEFAULT_Z_NEAR};

fn find_camera(
    parent_transform: &na::Transform3<f32>,
    current_node: &gltf::Node,
    resolution: &glm::Vec2,
) -> Option<Camera> {
    let current_transform = *parent_transform * trans_from_gltf(current_node.transform());
    if let Some(camera) = current_node.camera() {
        if let gltf::camera::Projection::Perspective(projection) = camera.projection() {
            let zfar = if let Some(far) = projection.zfar() {
                far
            } else {
                DEFAULT_Z_FAR
            };
            // TODO: maybe incorporate this in the future
            // let aspect_ratio = if let Some(aspect) = projection.aspect_ratio() {
            //     aspect
            // } else {
            //     std::f32::consts::FRAC_PI_2
            // };
            return Some(Camera::new(
                &na::try_convert(current_transform).unwrap(),
                &na::Perspective3::new(
                    resolution.x / resolution.y,
                    projection.yfov(),
                    projection.znear(),
                    zfar,
                ),
                &resolution,
            ));
        } else {
            for child in current_node.children() {
                return find_camera(&current_transform, &child, &resolution);
            }

            None
        }
    } else {
        for child in current_node.children() {
            return find_camera(&current_transform, &child, &resolution);
        }

        None
    }
}

pub fn get_camera(
    document: &gltf::Document,
    world_bound: &Bounds3,
    resolution: &glm::Vec2,
) -> Camera {
    let mut camera = get_default_camera(&world_bound, &resolution);
    'scene: for scene in document.scenes() {
        for node in scene.nodes() {
            if let Some(curr_cam) = find_camera(&na::Transform3::identity(), &node, &resolution) {
                camera = curr_cam;
                break 'scene;
            }
        }
    }

    camera
}

pub fn get_default_camera(world_bound: &Bounds3, resolution: &glm::Vec2) -> Camera {
    Camera::new(
        &na::Isometry3::look_at_rh(
            &world_bound.p_max,
            &na::Point3::origin(),
            &na::Vector3::new(0.0, 1.0, 0.0),
        )
        .inverse(),
        &na::Perspective3::new(
            resolution.x / resolution.y,
            std::f32::consts::FRAC_PI_2 * (resolution.y / resolution.x),
            DEFAULT_Z_NEAR,
            DEFAULT_Z_FAR,
        ),
        &resolution,
    )
}

pub fn trans_from_gltf(transform: gltf::scene::Transform) -> na::Projective3<f32> {
    let (translation, rotation, scaling) = transform.decomposed();

    let t = glm::translation(&glm::make_vec3(&translation));
    let r = glm::quat_to_mat4(&glm::make_quat(&rotation));
    let s = glm::scaling(&glm::make_vec3(&scaling));

    na::Projective3::from_matrix_unchecked(t * r * s)
}

pub fn from_gltf(
    log: &slog::Logger,
    path: &str,
    resolution: &na::Vector2<f32>,
    default_lights: bool,
) -> (
    Camera,
    crate::pathtracer::RenderScene,
    crate::viewer::ViewerScene,
) {
    let (document, buffers, images) = gltf::import(path).unwrap();
    let render_scene = crate::pathtracer::RenderScene::from_gltf(
        &log,
        &document,
        &buffers,
        &images,
        default_lights,
    );
    let camera = get_camera(&document, &render_scene.world_bound(), &resolution);
    let viewer_scene = crate::viewer::ViewerScene::from_gltf(&document, &buffers, &images);

    (camera, render_scene, viewer_scene)
}
