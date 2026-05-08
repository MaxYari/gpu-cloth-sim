# GPU Cloth — Path 3: Fully GPU via Texture2DRD + Custom Vertex Shader

## Goal

Eliminate the GPU→CPU readback entirely. Currently `_rd.sync()` + `buffer_get_data()` stalls the
game thread every frame waiting for the GPU, then hands the data back through `add_surface_from_arrays`
which reallocates the mesh's GPU vertex buffer from scratch. Path 3 removes all of that.

---

## Why the Current Approach Needs a Sync

The cloth sim uses a **local RenderingDevice** (`RenderingServer.create_local_rendering_device()`).
This is a completely separate GPU context from the main renderer. There is no way to share
a buffer between the two without a CPU round-trip. The sync+readback is structural.

---

## The Approach

### Core idea

Replace the mesh vertex buffer update with a **`Texture2DRD`** (a Godot texture backed by a
main-RD image) that the compute shader writes to, and a **custom vertex shader** that reads
positions/normals from that texture using `VERTEX_ID`. The renderer reads directly from the
texture the compute shader just wrote — zero CPU involvement.

### Why Texture2DRD instead of a buffer?

Godot's `mesh_surface_update_vertex_region` only accepts CPU bytes. There is no public API to
get the mesh's internal vertex buffer RID. If that RID were available, `RD.buffer_copy()` would
be the ideal path (GPU→GPU copy, hardware vertex fetch unit). Without it, textures are the only
way to feed compute output directly to a vertex shader within Godot's gdshader system.

An SSBO binding in the vertex stage would also work, but Godot's `spatial` gdshader does not
expose storage buffer uniforms in the vertex stage — you'd have to write raw GLSL draw pipelines
via RenderingDevice and lose Godot's material/lighting system.

---

## Frame Timing — No Delay

`call_on_render_thread` callables run **before** the render thread's scene draw calls in the
same frame they are submitted. The sequence is:

```
Game thread (frame N):
    _process():
        AnimationPlayer already ran (priority 0)
        Read bone matrices + collider transforms  ← correct, already animated
        Pack into PackedByteArray
        call_on_render_thread(lambda using those bytes)   ← submitted to this frame

Render thread (frame N):
    1. Run call_on_render_thread callable:
           buffer_update(bone_matrices)
           buffer_update(colliders)
           compute_list_begin → skin → PBD → update → compute_list_end
           RD.barrier(COMPUTE → VERTEX)  ← ensures writes visible before draw
    2. Draw scene: vertex shader samples positions texture  ← updated this frame
    3. Present
```

Bone matrices and collider transforms are captured on the game thread at the right time
(after AnimationPlayer) and handed to the callable as plain data. By the time the vertex
shader runs, the compute has already completed for the same frame. **No frame delay.**

This also means the game thread is never stalled — it submits work and immediately moves on
to frame N+1 while the render thread processes frame N.

---

## Per-Solver Texture Strategy

Each `GPUClothSolver` gets its **own** `Texture2DRD`, sized at `_initialize()` time when
`particle_count` is known. No shared global allocator, no coordination between solvers.

Each cloth mesh's `ShaderMaterial` has a different material instance with its own texture
uniform pointing to its solver's texture. The vertex shader samples only its own texture.

### Texture dimensions for arbitrary particle counts

A 1D texture hits driver limits (~16K max width). Use a 2D layout instead:

```gdscript
var tex_w := min(_particle_count, 4096)
var tex_h := ceili(float(_particle_count) / 4096.0)
```

In the compute shader: `imageStore(positions_img, ivec2(idx % 4096, idx / 4096), data)`
In the vertex shader: `texelFetch(positions_tex, ivec2(VERTEX_ID % 4096, VERTEX_ID / 4096), 0)`

This supports up to ~134M vertices per sim. Memory per sim at 1M vertices = 16MB for positions.
Across 100 sims, memory is proportional to actual usage — no pre-allocation waste.

### What if VRAM runs out?

With per-solver exact sizing, there is no artificial global limit. The only ceiling is VRAM,
which is the correct limit. If particle_count changes at runtime (unusual — mesh topology is
fixed at import), destroy and recreate the texture at the new size. This is expensive but rare.

---

## Pipeline Barrier Requirement

After the compute list ends and before the scene draws, Vulkan requires an explicit barrier
telling the GPU that compute writes to the texture are done and vertex reads may begin:

```gdscript
_rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE, RenderingDevice.BARRIER_MASK_VERTEX)
```

This is a cheap GPU-side synchronisation, not a CPU stall. Godot may insert automatic barriers
for Texture2DRD resources in some versions, but add the explicit call to be safe.

---

## Implementation Steps

### 1. Switch from local RD to main RD

Replace:
```gdscript
_rd = RenderingServer.create_local_rendering_device()
```
With:
```gdscript
_rd = RenderingServer.get_rendering_device()
```

All RD calls (`buffer_create`, `shader_create_from_spirv`, `compute_pipeline_create`,
`compute_list_begin`, etc.) now operate on the main RD.

**Critical**: on the main RD you do NOT call `_rd.submit()` or `_rd.sync()` — Godot's render
loop manages submission. All per-frame RD work must be wrapped in `call_on_render_thread`.

Buffer and shader creation (done once in `_initialize`) should also move to a render-thread
callable to ensure the main RD is in a valid state when accessed.

### 2. Create positions and normals as storage images

```gdscript
var fmt := RDTextureFormat.new()
fmt.format        = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT  # vec4
fmt.width         = tex_w
fmt.height        = tex_h
fmt.usage_bits    = (RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
                     RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
                     RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT)

var positions_img_rid := _rd.texture_create(fmt, RDTextureView.new())
var normals_img_rid   := _rd.texture_create(fmt, RDTextureView.new())
```

Wrap for the vertex shader:
```gdscript
var positions_texture := Texture2DRD.new()
positions_texture.texture_rd_rid = positions_img_rid

var normals_texture := Texture2DRD.new()
normals_texture.texture_rd_rid = normals_img_rid
```

Pass to the ShaderMaterial:
```gdscript
_cloth_material.set_shader_parameter("positions_tex", positions_texture)
_cloth_material.set_shader_parameter("normals_tex",   normals_texture)
```

### 3. Modify compute shaders

Change the final `cloth_update.glsl` output from writing to a storage buffer to writing to
a storage image:

```glsl
layout(set = 0, binding = 0, rgba32f) uniform writeonly image2D positions_img;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D normals_img;

// At the end of main():
ivec2 coord = ivec2(idx % 4096, idx / 4096);
imageStore(positions_img, coord, vec4(new_pos, w));
```

Add a normals compute pass (either a new `cloth_normals.glsl` or append to cloth_update):
- Pass 1: one thread per triangle → compute face normal → write to a per-triangle normal buffer
- Pass 2: one thread per vertex → sum normals from all touching triangles (using a build-time
  adjacency list: `vert_to_tris_buffer`) → normalize → imageStore to normals_img

The vertex→triangle adjacency list is built once in `_initialize()` from `_indices`.

### 4. Custom vertex shader

```glsl
shader_type spatial;

uniform sampler2D positions_tex : filter_nearest;
uniform sampler2D normals_tex   : filter_nearest;

void vertex() {
    ivec2 coord = ivec2(VERTEX_ID % 4096, VERTEX_ID / 4096);
    vec4  pos_data = texelFetch(positions_tex, coord, 0);
    vec3  sim_norm = texelFetch(normals_tex,   coord, 0).xyz;
    VERTEX = pos_data.xyz;
    NORMAL = sim_norm;
    // TANGENT can be derived from normal or also stored in a third texture
}
```

The output mesh is now a **static indexed mesh** built once from rest-pose positions. Godot
renders it (iterating over indices), but the vertex shader completely overrides `VERTEX` and
`NORMAL` from the textures. The mesh geometry is irrelevant for rendering; it only provides
the index buffer and draw call metadata.

### 5. AABB management

`mesh_surface_update_vertex_region` (and the texture approach) do not update the mesh AABB
used for frustum culling. Options:

- **Simplest**: set `_output_mesh_instance.extra_cull_margin = 10.0` so the mesh is never
  incorrectly culled when close to camera.
- **Better**: compute a rough AABB from skeleton bounds + max_travel_distance at init time,
  set `_output_mesh_instance.custom_aabb` once. Cloth never strays further than
  `max_travel_distance` from the skeleton, so a skeleton-AABB + padding is always correct.
- **Precise**: run a GPU min/max reduction pass to get tight bounds, readback only 2 vec3s
  per frame (very cheap compared to reading all vertex positions).

### 6. Remove the old output path

Once Path 3 is in place:
- Delete `_update_mesh()` and `_compute_tangents()` (CPU work, no longer needed)
- Delete `_positions_buffer` readback from `_simulate()`
- Delete `_rd.sync()` calls entirely
- `_output_mesh.add_surface_from_arrays` is called once at init with rest-pose geometry, never again

---

## Compute Normals — Two-Pass GPU Approach

The tricky part is that parallel normal accumulation has write conflicts (multiple triangles
write to the same vertex). Two solutions:

**Option A — Adjacency list (recommended)**

At init, build `vert_to_tri_offset[]` and `vert_to_tri_list[]` from `_indices`.
In the normals pass, each thread handles one vertex and iterates its triangle list:

```glsl
// cloth_normals.glsl
uint count  = vert_tri_counts[idx];
uint offset = vert_tri_offsets[idx];
vec3 normal = vec3(0.0);
for (uint t = 0u; t < count; t++) {
    normal += tri_normals[vert_tri_list[offset + t]];
}
normal = normalize(normal) * flip;
imageStore(normals_img, coord, vec4(normal, 0.0));
```

The per-triangle normals come from a first pass (one thread per triangle, no conflicts).

**Option B — Graph coloring (same as constraints)**

Apply greedy graph coloring to triangles (no two same-color triangles share a vertex),
dispatch each color group separately. More dispatch calls but no extra adjacency buffer.

---

## Summary: Current vs Path 3

| | Current | Path 3 |
|---|---|---|
| After compute | `rd.sync()` CPU stall | No stall |
| Positions to CPU | `buffer_get_data()` | Never |
| Normals/tangents | CPU loops | GPU compute pass |
| Mesh update | `clear_surfaces` + `add_surface_from_arrays` | None (static mesh) |
| CPU work per frame | Normals + tangents + mesh rebuild | Pack bone matrices only |
| Frame delay | None (sync is blocking) | None (callable runs before draw) |

---

## Prerequisite: Path 2 (recommended intermediate step)

Before Path 3, implement Path 2 (currently done as Path 1 quick win):
- Use `mesh_surface_update_vertex_region` instead of `clear_surfaces + add_surface_from_arrays`
- Move normals to a GPU compute pass, pack interleaved vertex buffer on GPU
- Single readback per frame but zero CPU math

Path 1/2 verify correctness of the oct-encoding and packing logic before the full refactor.
