// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Fine rasterizer. This can run in simple (just path rendering) and full
// modes, controllable by #define.

#import config
// @group(0) @binding(0)
// var<uniform> config: Config;

@group(0) @binding(0)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var texture1: texture_2d<f32>;

@group(0) @binding(2)
var texture2: texture_2d<f32>;

let PIXELS_PER_THREAD = 4u;

// The X size should be 16 / PIXELS_PER_THREAD
@compute @workgroup_size(4, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let xy = vec2(f32(global_id.x * PIXELS_PER_THREAD), f32(global_id.y));
    
    let xy_uint = vec2<u32>(xy);
    var rgba1: array<vec4<f32>, PIXELS_PER_THREAD>;
    var rgba2: array<vec4<f32>, PIXELS_PER_THREAD>;
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        //if coords.x < config.target_width && coords.y < config.target_height {
            rgba1[i] = vec4<f32>(0.,0.,0.,0.);//textureLoad(texture1, vec2<i32>(coords.xy), 0);
            rgba2[i] = textureLoad(texture2, vec2<i32>(coords.xy), 0); 
            var rgba_sep = vec4<f32>(0.,0.,0.,0.);
            if length(rgba1[i].rgb - rgba2[i].rgb) <= 0.01{
                rgba_sep = rgba1[i];
            }else if length(rgba1[i].rgb - vec3<f32>(0.,0.,0.)) <= 0.01{
                rgba_sep = rgba2[i];
            }else if length(rgba2[i].rgb - vec3<f32>(0.,0.,0.)) <= 0.01{
                rgba_sep = rgba1[i];
            }else{
                rgba_sep = (rgba1[i] + rgba2[i]) / 2.;
            }
            //let rgba_sep = length(rgba1[i], rgba2[i]) <= 0.1 ? rgba1[i] : (rgba1[i] + rgba2[i]) / 2.0;//rgba1[i] * rgba1[i].a + rgba2[i] * (1.0 - rgba1[i].a);    
            textureStore(output, vec2<i32>(coords), rgba_sep);
        //}
    }
}

