# Pathtracer-rs

This is a work in progress path tracer written in pure rust, using `nalgebra` for mathematical operations, `gltf` for GLTF scene file support, and `wgpu-rs` for frontend display.

This project heavily borrows from the book `Physically Based Rendering: From Theory to Implementation` by Matt Pharr et. al. which can be read online [here](http://www.pbr-book.org/).

## Current Renders

![render](https://user-images.githubusercontent.com/8923171/91656172-75625400-ea6b-11ea-9c71-015333cf244f.png)
*Render of the Sponza sample gltf scene, direct lighting only*