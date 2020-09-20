# Pathtracer-rs

![classroom_4096](https://user-images.githubusercontent.com/8923171/93718138-f96da000-fb2e-11ea-8354-a6fb34cd8bf2.png)

This is yet another path tracer written in rust, using `nalgebra` for mathematical operations, `gltf` for GLTF scene file support, and `wgpu-rs` for frontend display. It is primarily written for personal education and entertainment purposes, though it is also intended to be somewhat performant and capable of supporting multiple commonly used file formats

This project heavily borrows from the book `Physically Based Rendering: From Theory to Implementation` by Matt Pharr et. al. which can be read online [here](http://www.pbr-book.org/).

## Features

* Real time frontend preview for inspection and camera adjustments
* GLTF file format support (also supports the `KHR_lights_punctual`, `KHR_materials_ior`, and `KHR_materials_transmission` extensions, `KHR_materials_pbrSpecularGlossiness` support forthcoming)
* Mitsuba file format support (Work in progress, support is very ad hoc)
* Supported light types
  * Point Light
  * Directional Light
  * Area Light
  * Mesh Emission Map
  * Environmental Map
* Supported materials
  * Diffuse (Lambertian)
  * Metal
  * Pure Mirror
  * Glass
  * Substrate (Plastic in Mitsuba)
  * Microfacet model based on Torrance–Sparrow for metal, glass, and substrate materials
  * Disney BSDF (limited support)

## CLI Usage
```
cargo build --release
./target/release/pathtracer-rs --help
```

```
pathtracer_rs 1.0
Eric F. <eric1221bday@gmail.com>
Rust path tracer

USAGE:
    pathtracer-rs [FLAGS] [OPTIONS] <SCENE> --output <output>

FLAGS:
        --default_lights    Add default lights into the scene
    -h, --help              Prints help information
        --headless          run pathtracer in headless mode
    -V, --version           Prints version information

OPTIONS:
    -c, --camera <camera_controller>    Camera movement type [default: orbit]
    -l, --log_level <log_level>         Application wide log level [default: INFO]
    -d, --max_depth <max_depth>         Maximum ray tracing depth [default: 15]
    -m, --module_log <module_log>       Module names to log, (all for every module) [default: all]
    -o, --output <output>               Sets the output directory to save renders at
    -r, --resolution <resolution>       Resolution of the window
    -s, --samples <samples>             Number of samples path tracer to take per pixel (sampler dependent) [default: 1]

ARGS:
    <SCENE>    Sets the input scene to use
```

## Camera Controls

There are two camera control modes, first person and orbit, these are set using the CLI option `-c orbit` or `-c fp`

### Orbit Camera Controls

Camera view always points to the origin. Mouse click drag rotates camera about the origin. Mouse wheel moves the camera closer or further to the origin.

### First Person Camera Controls

* <kbd>W</kbd>/<kbd>A</kbd>/<kbd>S</kbd>/<kbd>D</kbd>: Moves the camera front, left, back, and right
* <kbd>Z</kbd>/<kbd>X</kbd>: Moves the camera up and down
* <kbd>Q</kbd>/<kbd>E</kbd>: Adjusts camera roll.
* Mouse click drag rotates the camera pitch and yaw in first person.

## Keyboard Shortcuts
* <kbd>R</kbd>: Renders image according to current camera and sampling settings
* <kbd>C</kbd>: Clears current render and returns to real time preview
* <kbd>CTRL</kbd>+<kbd>S</kbd>: Saves current rendered image to the directory specified in `--output` with name `render.png`
* <kbd>&#x2191;</kbd>/<kbd>&#x2193;</kbd>: Increases or decreases sample increment
* <kbd>CTRL</kbd>+<kbd>H</kbd>: Toggles displaying of mesh
* <kbd>CTRL</kbd>+<kbd>G</kbd>: Toggles displaying of wireframe outline

## Headless Mode

If the `--headless` flag is set, no preview window will be created. Rendering will proceed and the image will be saved at the `--output` directory with name `render.png` automatically.

## Future Work
* Subsurface Scattering
* Volume Rendering
* Path Guiding
* Wavefront MTL support
* SIMD based intersection routine with batched ray casting
* Optix based GPU acceleration structure and intersection routine
* GPU based integrator

## Gallery

### Sponza GLTF scene
![sponza_2048](https://user-images.githubusercontent.com/8923171/93720490-2f665080-fb3e-11ea-99e0-86657a535c70.png)
Can be found [here](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/Sponza).