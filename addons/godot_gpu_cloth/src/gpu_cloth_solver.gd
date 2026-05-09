@tool
class_name GPUClothSolver
extends Node3D

# ---------------------------------------------------------------------------
#  Exports
# ---------------------------------------------------------------------------
@export_group("Mesh Input")
## MeshInstance3D whose mesh and skeleton will drive the simulation.
@export var target_mesh: NodePath
## Skeleton3D that animates the mesh.
@export var skeleton: NodePath
## Which surface index on the ArrayMesh to simulate (0-based).
@export var surface_index: int = 0

@export_group("Physics")
@export var gravity_strength: float = -9.8
@export var solver_iterations: int = 8
@export var substeps: int = 8
@export var stiffness: float = 0.5
@export var damping: float = 0.99
@export var max_speed: float = 5.0
@export var max_travel_distance: float = 0.1

@export_group("Appearance")
## Must be a ShaderMaterial using cloth_surface.gdshader.
## If left empty a default ShaderMaterial is created automatically.
@export var cloth_material: Material
## Flip computed normals.
@export var flip_normals: bool = false

@export_group("Inertia")
@export var inertia_scale: Vector3 = Vector3(1.0, 1.0, 1.0)

@export_group("Wind")
@export var wind: Vector3 = Vector3.ZERO
@export var wind_turbulence: float = 0.3
@export var wind_frequency: float = 1.0

# ---------------------------------------------------------------------------
#  GPU resources  (all RIDs belong to the main RenderingDevice)
# ---------------------------------------------------------------------------
var _rd: RenderingDevice

# Simulation buffers
var _positions_buffer: RID
var _predicted_buffer: RID
var _velocities_buffer: RID
var _constraints_buffer: RID
var _colliders_buffer: RID

# Skinning buffers
var _rest_positions_buffer: RID
var _bone_indices_buffer: RID
var _bone_weights_skin_buffer: RID
var _bone_transforms_buffer: RID
var _skinned_targets_buffer: RID
var _cloth_weights_buffer: RID

# Normals / output buffers
var _face_normals_buffer: RID
var _indices_gpu_buffer: RID
var _vert_tri_counts_buffer: RID
var _vert_tri_offsets_buffer: RID
var _vert_tri_list_buffer: RID

# Output textures (compute writes → vertex shader reads)
var _positions_img_rid: RID
var _normals_img_rid: RID
var _positions_tex: Texture2DRD
var _normals_tex: Texture2DRD
var _tex_w: int
var _tex_h: int

# Shaders
var _skin_shader: RID
var _predict_shader: RID
var _solve_shader: RID
var _update_shader: RID
var _collide_shader: RID
var _warm_start_shader: RID
var _normals_shader: RID
var _output_shader: RID

# Pipelines
var _skin_pipeline: RID
var _predict_pipeline: RID
var _solve_pipeline: RID
var _update_pipeline: RID
var _collide_pipeline: RID
var _warm_start_pipeline: RID
var _normals_pipeline: RID
var _output_pipeline: RID

# Uniform sets
var _skin_uniform_set: RID
var _predict_uniform_set: RID
var _solve_uniform_set: RID
var _update_uniform_set: RID
var _collide_uniform_set: RID
var _warm_start_uniform_set: RID
var _normals_uniform_set: RID
var _output_uniform_set: RID

# ---------------------------------------------------------------------------
#  Runtime state
# ---------------------------------------------------------------------------
var _mesh_instance_node: MeshInstance3D
var _skeleton_node: Skeleton3D
var _skin: Skin

var _particle_count: int
var _tri_count: int
var _constraint_count: int
var _constraint_groups: Array = []
var _bind_count: int
var _bind_to_bone: PackedInt32Array

var _colliders: Array[GPUClothCollider] = []
var _collider_count: int = 0

var _uvs: PackedVector2Array
var _indices: PackedInt32Array

var _output_mesh: ArrayMesh
var _output_mesh_instance: MeshInstance3D
var _cloth_mat: ShaderMaterial   # always our ShaderMaterial

var _prev_mesh_world_pos: Vector3

# Reusable push constant buffers (built once, mutated per-frame on game thread)
var _skin_push: PackedByteArray
var _pbd_push: PackedByteArray
var _output_push: PackedByteArray  # constant after init

var _plugin_dir: String
var _gpu_init_done := false
var _needs_warm_start := true


# ---------------------------------------------------------------------------
#  Lifecycle
# ---------------------------------------------------------------------------

func _ready() -> void:
	_plugin_dir = get_script().resource_path.get_base_dir().get_base_dir()

	if Engine.is_editor_hint():
		return

	if not RenderingServer.get_rendering_device():
		push_error("[GPUCloth] Requires Vulkan renderer (Forward+ or Mobile). Aborting.")
		return

	set_process_priority(100)
	print("[GPUCloth] ── Initializing GPUClothSolver ──────────────────────────")
	_initialize()


func _process(delta: float) -> void:
	if Engine.is_editor_hint() or not _gpu_init_done:
		return
	if _needs_warm_start:
		RenderingServer.call_on_render_thread(_gpu_do_warm_start)
		_needs_warm_start = false
		return
	_simulate(delta)


func _exit_tree() -> void:
	if not _gpu_init_done:
		return
	_gpu_init_done = false
	# Capture all RIDs so the lambda stays safe after this node is freed.
	var rids: Array = [
		_positions_buffer, _predicted_buffer, _velocities_buffer,
		_constraints_buffer, _colliders_buffer,
		_rest_positions_buffer, _bone_indices_buffer, _bone_weights_skin_buffer,
		_bone_transforms_buffer, _skinned_targets_buffer, _cloth_weights_buffer,
		_face_normals_buffer, _indices_gpu_buffer,
		_vert_tri_counts_buffer, _vert_tri_offsets_buffer, _vert_tri_list_buffer,
		_positions_img_rid, _normals_img_rid,
		_skin_uniform_set, _predict_uniform_set, _solve_uniform_set,
		_update_uniform_set, _collide_uniform_set, _warm_start_uniform_set,
		_normals_uniform_set, _output_uniform_set,
		_skin_pipeline, _predict_pipeline, _solve_pipeline,
		_update_pipeline, _collide_pipeline, _warm_start_pipeline,
		_normals_pipeline, _output_pipeline,
		_skin_shader, _predict_shader, _solve_shader,
		_update_shader, _collide_shader, _warm_start_shader,
		_normals_shader, _output_shader,
	]
	RenderingServer.call_on_render_thread(func() -> void:
		var rd := RenderingServer.get_rendering_device()
		if not rd:
			return
		for rid: RID in rids:
			if rid.is_valid():
				rd.free_rid(rid)
	)
	print("[GPUCloth] GPU resource cleanup queued on render thread.")


# ---------------------------------------------------------------------------
#  Initialization  (game thread — CPU work only)
# ---------------------------------------------------------------------------

func _initialize() -> void:
	# ── Resolve nodes ───────────────────────────────────────────────────────
	_mesh_instance_node = get_node_or_null(target_mesh) as MeshInstance3D
	if not _mesh_instance_node:
		push_error("[GPUCloth] 'target_mesh' is not set or not a MeshInstance3D.")
		return

	_skeleton_node = get_node_or_null(skeleton) as Skeleton3D
	if not _skeleton_node:
		push_error("[GPUCloth] 'skeleton' is not set or not a Skeleton3D.")
		return
	print("[GPUCloth] Skeleton: %s  bones: %d" % [_skeleton_node.name, _skeleton_node.get_bone_count()])

	_skin = _mesh_instance_node.get_skin()
	if not _skin:
		push_error("[GPUCloth] target_mesh has no Skin resource.")
		return
	_bind_count = _skin.get_bind_count()
	print("[GPUCloth] Skin binds: %d" % _bind_count)

	# ── Read ArrayMesh surface ───────────────────────────────────────────────
	var arr_mesh := _mesh_instance_node.mesh as ArrayMesh
	if not arr_mesh or arr_mesh.get_surface_count() == 0:
		push_error("[GPUCloth] target_mesh has no ArrayMesh surfaces.")
		return
	if surface_index >= arr_mesh.get_surface_count():
		push_error("[GPUCloth] surface_index=%d but mesh only has %d surface(s)." \
			% [surface_index, arr_mesh.get_surface_count()])
		return
	print("[GPUCloth] Simulating surface %d of %d." % [surface_index, arr_mesh.get_surface_count()])

	var surf := arr_mesh.surface_get_arrays(surface_index)
	var vert_arr: PackedVector3Array    = surf[Mesh.ARRAY_VERTEX]  if surf[Mesh.ARRAY_VERTEX]  else PackedVector3Array()
	var color_arr: PackedColorArray     = surf[Mesh.ARRAY_COLOR]   if surf[Mesh.ARRAY_COLOR]   else PackedColorArray()
	var bones_raw: PackedInt32Array     = surf[Mesh.ARRAY_BONES]   if surf[Mesh.ARRAY_BONES]   else PackedInt32Array()
	var weights_raw: PackedFloat32Array = surf[Mesh.ARRAY_WEIGHTS] if surf[Mesh.ARRAY_WEIGHTS] else PackedFloat32Array()
	_uvs     = surf[Mesh.ARRAY_TEX_UV] if surf[Mesh.ARRAY_TEX_UV] else PackedVector2Array()
	_indices = surf[Mesh.ARRAY_INDEX]  if surf[Mesh.ARRAY_INDEX]  else PackedInt32Array()

	_particle_count = vert_arr.size()
	_tri_count      = _indices.size() / 3
	print("[GPUCloth] Vertices: %d  Triangles: %d" % [_particle_count, _tri_count])

	if _indices.is_empty():
		push_error("[GPUCloth] Mesh has no index array. The mesh must be indexed.")
		return
	if color_arr.is_empty():
		push_error("[GPUCloth] Mesh has no vertex color (ARRAY_COLOR). Paint cloth_weight in Blender.")
		return
	if bones_raw.is_empty() or weights_raw.is_empty():
		push_error("[GPUCloth] Mesh has no bone data. The mesh must be skinned.")
		return

	# ── Cloth weights from vertex color red channel ──────────────────────────
	var cloth_weights := PackedFloat32Array()
	cloth_weights.resize(_particle_count)
	var n_anchored := 0; var n_blend := 0; var n_free := 0
	for i in _particle_count:
		var cw: float = clamp(color_arr[i].r, 0.0, 1.0)
		cloth_weights[i] = cw
		if   cw < 0.01:  n_anchored += 1
		elif cw > 0.99:  n_free     += 1
		else:             n_blend    += 1
	print("[GPUCloth] Cloth weights → anchored: %d  blend: %d  free: %d" % [n_anchored, n_blend, n_free])
	if n_anchored == 0:
		push_warning("[GPUCloth] Zero anchored vertices! Cloth will fall freely.")

	# ── Resolve bind → skeleton bone index ──────────────────────────────────
	_bind_to_bone.resize(_bind_count)
	for bi in _bind_count:
		var bone_idx: int = _skin.get_bind_bone(bi)
		if bone_idx < 0:
			bone_idx = _skeleton_node.find_bone(str(_skin.get_bind_name(bi)))
		_bind_to_bone[bi] = bone_idx

	# ── Build position buffers ───────────────────────────────────────────────
	var mesh_to_skel := _skeleton_node.global_transform.affine_inverse() \
		* _mesh_instance_node.global_transform

	var pos_data  := PackedFloat32Array(); pos_data.resize(_particle_count * 4)
	var rest_data := PackedFloat32Array(); rest_data.resize(_particle_count * 4)

	for i in _particle_count:
		var skel_pos: Vector3 = mesh_to_skel * vert_arr[i]
		var inv_mass: float   = 0.0 if cloth_weights[i] < 0.01 else 1.0
		pos_data[i*4+0] = skel_pos.x; pos_data[i*4+1] = skel_pos.y
		pos_data[i*4+2] = skel_pos.z; pos_data[i*4+3] = inv_mass
		rest_data[i*4+0] = vert_arr[i].x; rest_data[i*4+1] = vert_arr[i].y
		rest_data[i*4+2] = vert_arr[i].z; rest_data[i*4+3] = 1.0

	# ── Bone index buffer ────────────────────────────────────────────────────
	var bone_idx_bytes := PackedByteArray(); bone_idx_bytes.resize(_particle_count * 8)
	for i in _particle_count:
		var b := i * 4
		bone_idx_bytes.encode_u32(i*8+0, (bones_raw[b] & 0xFFFF) | ((bones_raw[b+1] & 0xFFFF) << 16))
		bone_idx_bytes.encode_u32(i*8+4, (bones_raw[b+2] & 0xFFFF) | ((bones_raw[b+3] & 0xFFFF) << 16))

	# ── Bone weight buffer ───────────────────────────────────────────────────
	var bone_w_data := PackedFloat32Array(); bone_w_data.resize(_particle_count * 4)
	for i in _particle_count:
		var b := i * 4
		bone_w_data[b+0] = weights_raw[b+0]; bone_w_data[b+1] = weights_raw[b+1]
		bone_w_data[b+2] = weights_raw[b+2]; bone_w_data[b+3] = weights_raw[b+3]

	# ── Cloth weights buffer ─────────────────────────────────────────────────
	var cloth_w_data := PackedFloat32Array(); cloth_w_data.resize(_particle_count * 4)
	for i in _particle_count:
		cloth_w_data[i*4+0] = cloth_weights[i]

	# ── Build constraints ────────────────────────────────────────────────────
	var con_data := _build_constraints(vert_arr)
	_constraint_count = con_data.size() / 4
	print("[GPUCloth] Constraints: %d in %d groups" % [_constraint_count, _constraint_groups.size()])

	# ── Discover colliders ───────────────────────────────────────────────────
	_colliders.clear()
	_find_colliders_recursive(_skeleton_node)
	_collider_count = _colliders.size()
	print("[GPUCloth] Colliders: %d" % _collider_count)

	# ── Build vertex→triangle adjacency list ─────────────────────────────────
	var adj := _build_adjacency()

	# ── Texture dimensions ───────────────────────────────────────────────────
	_tex_w = mini(_particle_count, 4096)
	_tex_h = ceili(float(_particle_count) / float(_tex_w))
	print("[GPUCloth] Output texture: %dx%d" % [_tex_w, _tex_h])

	# ── Texture2DRD objects (game thread is fine; RIDs assigned on render thread)
	_positions_tex = Texture2DRD.new()
	_normals_tex   = Texture2DRD.new()

	# ── Output mesh (built once from rest-pose, vertex shader overrides VERTEX/NORMAL)
	_build_output_mesh(vert_arr, mesh_to_skel)

	# ── Push constant buffers ────────────────────────────────────────────────
	_skin_push = PackedByteArray(); _skin_push.resize(64)
	_skin_push.encode_u32(0, _particle_count)
	_skin_push.encode_u32(4, _bind_count)

	_pbd_push = PackedByteArray(); _pbd_push.resize(64)

	_output_push = PackedByteArray(); _output_push.resize(16)
	_output_push.encode_u32(0, _particle_count)
	_output_push.encode_u32(4, _tex_w)

	# ── Pack init data for the render-thread callable ────────────────────────
	var init_data := {
		"pos_bytes":         pos_data.to_byte_array(),
		"rest_bytes":        rest_data.to_byte_array(),
		"vel_bytes":         PackedByteArray(), # zero-initialised below
		"con_bytes":         con_data.to_byte_array(),
		"bone_w_bytes":      bone_w_data.to_byte_array(),
		"cloth_w_bytes":     cloth_w_data.to_byte_array(),
		"bone_idx_bytes":    bone_idx_bytes,
		"bone_mat_bytes":    _pack_bone_matrices(),
		"col_bytes":         _pack_colliders(),
		"idx_bytes":         _pack_indices_uint(),
		"adj_counts_bytes":  adj.counts.to_byte_array(),
		"adj_offsets_bytes": adj.offsets.to_byte_array(),
		"adj_list_bytes":    adj.list.to_byte_array(),
		"adj_list_size":     adj.list.size(),
	}
	# zero-filled velocity buffer
	init_data["vel_bytes"].resize(_particle_count * 16)

	RenderingServer.call_on_render_thread(_gpu_do_init.bind(init_data))
	print("[GPUCloth] GPU init queued on render thread.")

	_prev_mesh_world_pos = _skeleton_node.global_position
	print("[GPUCloth] ── CPU initialization complete ───────────────────────────")


# ---------------------------------------------------------------------------
#  Build static output mesh  (called once; vertex shader overrides geometry)
# ---------------------------------------------------------------------------

func _build_output_mesh(vert_arr: PackedVector3Array, mesh_to_skel: Transform3D) -> void:
	_output_mesh          = ArrayMesh.new()
	_output_mesh_instance = MeshInstance3D.new()
	_output_mesh_instance.mesh = _output_mesh
	_output_mesh_instance.name = "GPUClothOutput"

	# Convert rest positions to skel-local space (solver space).
	var verts := PackedVector3Array(); verts.resize(_particle_count)
	for i in _particle_count:
		verts[i] = mesh_to_skel * vert_arr[i]

	# Dummy normals — vertex shader overrides these every frame.
	var dummy_normals := PackedVector3Array(); dummy_normals.resize(_particle_count)
	dummy_normals.fill(Vector3.UP)

	var arrays: Array = []; arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = dummy_normals
	arrays[Mesh.ARRAY_INDEX]  = _indices
	if not _uvs.is_empty():
		arrays[Mesh.ARRAY_TEX_UV] = _uvs

	_output_mesh.clear_surfaces()
	_output_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)

	# Generous cull margin so cloth is never incorrectly frustum-culled.
	_output_mesh_instance.extra_cull_margin = 10.0

	# Material: must use cloth_surface.gdshader for the vertex stage.
	if cloth_material is ShaderMaterial:
		_cloth_mat = cloth_material as ShaderMaterial
	else:
		if cloth_material:
			push_warning("[GPUCloth] cloth_material must be a ShaderMaterial using " +
				"cloth_surface.gdshader. Creating a default one.")
		var shader: Shader = load(_plugin_dir + "/shaders/cloth_surface.gdshader")
		_cloth_mat = ShaderMaterial.new()
		_cloth_mat.shader = shader

	_output_mesh_instance.material_override = _cloth_mat
	_cloth_mat.set_shader_parameter("tex_width", _tex_w)
	# positions_tex / normals_tex are set after the render thread creates the image RIDs.
	_cloth_mat.set_shader_parameter("positions_tex", _positions_tex)
	_cloth_mat.set_shader_parameter("normals_tex",   _normals_tex)

	_skeleton_node.add_child(_output_mesh_instance)
	_output_mesh_instance.transform = Transform3D.IDENTITY
	print("[GPUCloth] Output mesh added as child of '%s'." % _skeleton_node.name)

	# Hide the simulated surface on the original mesh.
	var invisible_mat := StandardMaterial3D.new()
	invisible_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	invisible_mat.albedo_color  = Color(0, 0, 0, 0)
	invisible_mat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	_mesh_instance_node.set_surface_override_material(surface_index, invisible_mat)


# ---------------------------------------------------------------------------
#  GPU init  (render thread)
# ---------------------------------------------------------------------------

func _gpu_do_init(init_data: Dictionary) -> void:
	_rd = RenderingServer.get_rendering_device()

	var pos_bytes:         PackedByteArray = init_data["pos_bytes"]
	var rest_bytes:        PackedByteArray = init_data["rest_bytes"]
	var vel_bytes:         PackedByteArray = init_data["vel_bytes"]
	var con_bytes:         PackedByteArray = init_data["con_bytes"]
	var bone_w_bytes:      PackedByteArray = init_data["bone_w_bytes"]
	var cloth_w_bytes:     PackedByteArray = init_data["cloth_w_bytes"]
	var bone_idx_bytes:    PackedByteArray = init_data["bone_idx_bytes"]
	var bone_mat_bytes:    PackedByteArray = init_data["bone_mat_bytes"]
	var col_bytes:         PackedByteArray = init_data["col_bytes"]
	var idx_bytes:         PackedByteArray = init_data["idx_bytes"]
	var adj_counts_bytes:  PackedByteArray = init_data["adj_counts_bytes"]
	var adj_offsets_bytes: PackedByteArray = init_data["adj_offsets_bytes"]
	var adj_list_bytes:    PackedByteArray = init_data["adj_list_bytes"]
	var adj_list_size:     int             = init_data["adj_list_size"]

	# ── Storage buffers ──────────────────────────────────────────────────────
	_positions_buffer         = _rd.storage_buffer_create(pos_bytes.size(),       pos_bytes)
	_predicted_buffer         = _rd.storage_buffer_create(pos_bytes.size(),       pos_bytes)
	_velocities_buffer        = _rd.storage_buffer_create(vel_bytes.size(),       vel_bytes)
	_constraints_buffer       = _rd.storage_buffer_create(max(con_bytes.size(), 64), con_bytes)
	_colliders_buffer         = _rd.storage_buffer_create(max(col_bytes.size(), 64), col_bytes)
	_rest_positions_buffer    = _rd.storage_buffer_create(rest_bytes.size(),      rest_bytes)
	_bone_indices_buffer      = _rd.storage_buffer_create(bone_idx_bytes.size(),  bone_idx_bytes)
	_bone_weights_skin_buffer = _rd.storage_buffer_create(bone_w_bytes.size(),    bone_w_bytes)
	_bone_transforms_buffer   = _rd.storage_buffer_create(max(bone_mat_bytes.size(), 64), bone_mat_bytes)
	_cloth_weights_buffer     = _rd.storage_buffer_create(cloth_w_bytes.size(),   cloth_w_bytes)
	_skinned_targets_buffer   = _rd.storage_buffer_create(pos_bytes.size(),       pos_bytes)

	_face_normals_buffer      = _rd.storage_buffer_create(max(_tri_count * 16, 64))
	_indices_gpu_buffer       = _rd.storage_buffer_create(idx_bytes.size(),       idx_bytes)
	_vert_tri_counts_buffer   = _rd.storage_buffer_create(adj_counts_bytes.size(),  adj_counts_bytes)
	_vert_tri_offsets_buffer  = _rd.storage_buffer_create(adj_offsets_bytes.size(), adj_offsets_bytes)
	_vert_tri_list_buffer     = _rd.storage_buffer_create(max(adj_list_bytes.size(), 16), adj_list_bytes)

	# ── Output storage images ────────────────────────────────────────────────
	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.width      = _tex_w
	fmt.height     = _tex_h
	fmt.usage_bits = (RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
					  RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
					  RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT)

	_positions_img_rid = _rd.texture_create(fmt, RDTextureView.new())
	_normals_img_rid   = _rd.texture_create(fmt, RDTextureView.new())

	# Wire the images into the Texture2DRD wrappers so the vertex shader can sample them.
	_positions_tex.texture_rd_rid = _positions_img_rid
	_normals_tex.texture_rd_rid   = _normals_img_rid

	# ── Shaders ──────────────────────────────────────────────────────────────
	_skin_shader        = _load_shader(_plugin_dir + "/shaders/compute/cloth_skin.glsl")
	_predict_shader     = _load_shader(_plugin_dir + "/shaders/compute/cloth_predict.glsl")
	_solve_shader       = _load_shader(_plugin_dir + "/shaders/compute/cloth_solve.glsl")
	_update_shader      = _load_shader(_plugin_dir + "/shaders/compute/cloth_update.glsl")
	_collide_shader     = _load_shader(_plugin_dir + "/shaders/compute/cloth_collide.glsl")
	_warm_start_shader  = _load_shader(_plugin_dir + "/shaders/compute/cloth_warm_start.glsl")
	_normals_shader     = _load_shader(_plugin_dir + "/shaders/compute/cloth_normals.glsl")
	_output_shader      = _load_shader(_plugin_dir + "/shaders/compute/cloth_output.glsl")

	# ── Pipelines ────────────────────────────────────────────────────────────
	_skin_pipeline       = _rd.compute_pipeline_create(_skin_shader)
	_predict_pipeline    = _rd.compute_pipeline_create(_predict_shader)
	_solve_pipeline      = _rd.compute_pipeline_create(_solve_shader)
	_update_pipeline     = _rd.compute_pipeline_create(_update_shader)
	_collide_pipeline    = _rd.compute_pipeline_create(_collide_shader)
	_warm_start_pipeline = _rd.compute_pipeline_create(_warm_start_shader)
	_normals_pipeline    = _rd.compute_pipeline_create(_normals_shader)
	_output_pipeline     = _rd.compute_pipeline_create(_output_shader)

	# ── Uniform sets ──────────────────────────────────────────────────────────
	_skin_uniform_set = _create_uniform_set(_skin_shader, [
		_make_uniform(0, _rest_positions_buffer),
		_make_uniform(1, _bone_indices_buffer),
		_make_uniform(2, _bone_weights_skin_buffer),
		_make_uniform(3, _bone_transforms_buffer),
		_make_uniform(4, _skinned_targets_buffer),
	])
	_predict_uniform_set = _create_uniform_set(_predict_shader, [
		_make_uniform(0, _positions_buffer),
		_make_uniform(1, _predicted_buffer),
		_make_uniform(2, _velocities_buffer),
		_make_uniform(5, _skinned_targets_buffer),
	])
	_solve_uniform_set = _create_uniform_set(_solve_shader, [
		_make_uniform(1, _predicted_buffer),
		_make_uniform(3, _constraints_buffer),
	])
	_update_uniform_set = _create_uniform_set(_update_shader, [
		_make_uniform(0, _positions_buffer),
		_make_uniform(1, _predicted_buffer),
		_make_uniform(2, _velocities_buffer),
		_make_uniform(5, _cloth_weights_buffer),
		_make_uniform(6, _skinned_targets_buffer),
	])
	_collide_uniform_set = _create_uniform_set(_collide_shader, [
		_make_uniform(1, _predicted_buffer),
		_make_uniform(4, _colliders_buffer),
		_make_uniform(5, _skinned_targets_buffer),
	])
	_warm_start_uniform_set = _create_uniform_set(_warm_start_shader, [
		_make_uniform(0, _positions_buffer),
		_make_uniform(1, _predicted_buffer),
		_make_uniform(2, _velocities_buffer),
		_make_uniform(4, _skinned_targets_buffer),
	])
	_normals_uniform_set = _create_uniform_set(_normals_shader, [
		_make_uniform(0, _positions_buffer),
		_make_uniform(1, _indices_gpu_buffer),
		_make_uniform(2, _face_normals_buffer),
	])
	_output_uniform_set = _create_uniform_set(_output_shader, [
		_make_uniform(0, _positions_buffer),
		_make_uniform(1, _face_normals_buffer),
		_make_uniform(2, _vert_tri_counts_buffer),
		_make_uniform(3, _vert_tri_offsets_buffer),
		_make_uniform(4, _vert_tri_list_buffer),
		_make_image_uniform(5, _positions_img_rid),
		_make_image_uniform(6, _normals_img_rid),
	])

	_gpu_init_done = true
	print("[GPUCloth] GPU init complete on render thread.")


# ---------------------------------------------------------------------------
#  Warm start  (render thread) — skin pass then copy to positions/predicted
# ---------------------------------------------------------------------------

func _gpu_do_warm_start() -> void:
	var groups := ceili(float(_particle_count) / 64.0)
	var cl := _rd.compute_list_begin()

	_rd.compute_list_bind_compute_pipeline(cl, _skin_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _skin_uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, _skin_push, 64)
	_rd.compute_list_dispatch(cl, groups, 1, 1)
	_rd.compute_list_add_barrier(cl)

	var wp := PackedByteArray(); wp.resize(16)
	wp.encode_u32(0, _particle_count)
	_rd.compute_list_bind_compute_pipeline(cl, _warm_start_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _warm_start_uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, wp, 16)
	_rd.compute_list_dispatch(cl, groups, 1, 1)
	_rd.compute_list_add_barrier(cl)

	# Initialise output textures so the mesh looks correct from frame 1.
	_dispatch_output_passes(cl, groups, 1.0 if not flip_normals else -1.0)

	_rd.compute_list_end()
	_rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE, RenderingDevice.BARRIER_MASK_VERTEX)
	print("[GPUCloth] Warm start complete on render thread.")


# ---------------------------------------------------------------------------
#  Per-frame simulation  (game thread packs data, render thread dispatches)
# ---------------------------------------------------------------------------

func _simulate(delta: float) -> void:
	var sub_dt := delta / float(substeps)

	# ── Pack bone matrices and collider data on the game thread ───────────────
	var bone_bytes := _pack_bone_matrices()
	var col_bytes  := _pack_colliders() if _collider_count > 0 else PackedByteArray()

	# ── Inertia ───────────────────────────────────────────────────────────────
	var delta_world := _skeleton_node.global_position - _prev_mesh_world_pos
	var delta_local := _skeleton_node.global_transform.basis.inverse() * delta_world
	var inertia_sub := delta_local * inertia_scale / float(substeps)
	_prev_mesh_world_pos = _skeleton_node.global_position

	# ── Wind ──────────────────────────────────────────────────────────────────
	var t   := Time.get_ticks_msec() / 1000.0 * wind_frequency
	var gust := Vector3(
		sin(t * 1.7) + sin(t * 3.1 + 1.3),
		sin(t * 1.3 + 2.0) + sin(t * 2.7 + 0.7),
		sin(t * 2.1 + 4.0) + sin(t * 1.9 + 3.1)) * 0.5
	var eff_wind  := wind + wind.length() * gust * wind_turbulence
	var local_wind := global_transform.basis.inverse() * eff_wind

	# ── Base PBD push constant ─────────────────────────────────────────────────
	_pbd_push.encode_float(0,  sub_dt)
	_pbd_push.encode_float(4,  gravity_strength)
	_pbd_push.encode_u32(8,    _particle_count)
	_pbd_push.encode_u32(12,   _constraint_count)
	_pbd_push.encode_float(16, damping)
	_pbd_push.encode_float(20, max_speed)
	_pbd_push.encode_u32(24,   _collider_count)
	_pbd_push.encode_u32(28,   0)
	_pbd_push.encode_float(32, inertia_sub.x)
	_pbd_push.encode_float(36, inertia_sub.y)
	_pbd_push.encode_float(40, inertia_sub.z)
	_pbd_push.encode_float(44, max_travel_distance)
	_pbd_push.encode_float(48, local_wind.x)
	_pbd_push.encode_float(52, local_wind.y)
	_pbd_push.encode_float(56, local_wind.z)
	_pbd_push.encode_float(60, 1.0 / float(substeps))

	# Capture by value — the render thread callable mutates this copy.
	var push_copy      := _pbd_push.duplicate()
	var nflip          := -1.0 if flip_normals else 1.0
	var cap_substeps   := substeps
	var cap_iters      := solver_iterations

	RenderingServer.call_on_render_thread(
		_gpu_do_simulate.bind(bone_bytes, col_bytes, push_copy, nflip, cap_substeps, cap_iters))


func _gpu_do_simulate(
		bone_bytes: PackedByteArray,
		col_bytes:  PackedByteArray,
		push:       PackedByteArray,
		nflip:      float,
		p_substeps: int,
		p_iters:    int) -> void:

	_rd.buffer_update(_bone_transforms_buffer, 0, bone_bytes.size(), bone_bytes)
	if col_bytes.size() > 0:
		_rd.buffer_update(_colliders_buffer, 0, col_bytes.size(), col_bytes)

	var groups := ceili(float(_particle_count) / 64.0)
	var cl     := _rd.compute_list_begin()

	# SKIN
	_rd.compute_list_bind_compute_pipeline(cl, _skin_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _skin_uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, _skin_push, 64)
	_rd.compute_list_dispatch(cl, groups, 1, 1)
	_rd.compute_list_add_barrier(cl)

	for _s in p_substeps:
		# PREDICT
		_rd.compute_list_bind_compute_pipeline(cl, _predict_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _predict_uniform_set, 0)
		_rd.compute_list_set_push_constant(cl, push, 64)
		_rd.compute_list_dispatch(cl, groups, 1, 1)
		_rd.compute_list_add_barrier(cl)

		# SOLVE + COLLIDE interleaved
		_rd.compute_list_bind_compute_pipeline(cl, _solve_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _solve_uniform_set, 0)
		for _iter in p_iters:
			for grp in _constraint_groups:
				push.encode_u32(12, grp.count)
				push.encode_u32(28, grp.offset)
				_rd.compute_list_set_push_constant(cl, push, 64)
				_rd.compute_list_dispatch(cl, ceili(float(grp.count) / 64.0), 1, 1)
				_rd.compute_list_add_barrier(cl)

			if _collider_count > 0:
				push.encode_u32(12, _constraint_count)
				push.encode_u32(28, 0)
				_rd.compute_list_bind_compute_pipeline(cl, _collide_pipeline)
				_rd.compute_list_bind_uniform_set(cl, _collide_uniform_set, 0)
				_rd.compute_list_set_push_constant(cl, push, 64)
				_rd.compute_list_dispatch(cl, groups, 1, 1)
				_rd.compute_list_add_barrier(cl)
				_rd.compute_list_bind_compute_pipeline(cl, _solve_pipeline)
				_rd.compute_list_bind_uniform_set(cl, _solve_uniform_set, 0)

		# UPDATE
		push.encode_u32(12, _constraint_count)
		push.encode_u32(28, 0)
		_rd.compute_list_bind_compute_pipeline(cl, _update_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _update_uniform_set, 0)
		_rd.compute_list_set_push_constant(cl, push, 64)
		_rd.compute_list_dispatch(cl, groups, 1, 1)
		_rd.compute_list_add_barrier(cl)

	# NORMALS + OUTPUT → write positions_img and normals_img
	_dispatch_output_passes(cl, groups, nflip)

	_rd.compute_list_end()
	_rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE, RenderingDevice.BARRIER_MASK_VERTEX)


# Shared helper: dispatch the normals face pass then the output vertex pass.
func _dispatch_output_passes(cl: int, groups: int, nflip: float) -> void:
	var tri_groups := ceili(float(_tri_count) / 64.0)

	var np := PackedByteArray(); np.resize(16)
	np.encode_u32(0, _tri_count)
	np.encode_float(4, nflip)

	_rd.compute_list_bind_compute_pipeline(cl, _normals_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _normals_uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, np, 16)
	_rd.compute_list_dispatch(cl, tri_groups, 1, 1)
	_rd.compute_list_add_barrier(cl)

	_rd.compute_list_bind_compute_pipeline(cl, _output_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _output_uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, _output_push, 16)
	_rd.compute_list_dispatch(cl, groups, 1, 1)
	# Caller adds the final barrier / compute_list_end.


# ---------------------------------------------------------------------------
#  Adjacency list  (vertex → touching triangles)
# ---------------------------------------------------------------------------

func _build_adjacency() -> Dictionary:
	var vert_tris: Array = []
	vert_tris.resize(_particle_count)
	for v in _particle_count:
		vert_tris[v] = PackedInt32Array()
	for t in _tri_count:
		vert_tris[_indices[t*3+0]].append(t)
		vert_tris[_indices[t*3+1]].append(t)
		vert_tris[_indices[t*3+2]].append(t)

	var counts  := PackedInt32Array(); counts.resize(_particle_count)
	var offsets := PackedInt32Array(); offsets.resize(_particle_count)
	var list    := PackedInt32Array()
	var offset  := 0
	for v in _particle_count:
		counts[v]  = vert_tris[v].size()
		offsets[v] = offset
		list.append_array(vert_tris[v])
		offset += vert_tris[v].size()

	print("[GPUCloth] Adjacency list: %d entries for %d vertices." % [list.size(), _particle_count])
	return {counts = counts, offsets = offsets, list = list}


# ---------------------------------------------------------------------------
#  Triangle index buffer (uint32)
# ---------------------------------------------------------------------------

func _pack_indices_uint() -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(_indices.size() * 4)
	for i in _indices.size():
		bytes.encode_u32(i * 4, _indices[i])
	return bytes


# ---------------------------------------------------------------------------
#  Constraint building  (unchanged)
# ---------------------------------------------------------------------------

func _build_constraints(verts: PackedVector3Array) -> PackedFloat32Array:
	print("[GPUCloth] Building constraints from %d triangles..." % _tri_count)

	var edge_set: Dictionary = {}
	for t in _tri_count:
		var ia := _indices[t*3+0]; var ib := _indices[t*3+1]; var ic := _indices[t*3+2]
		for pair in [[ia, ib], [ib, ic], [ia, ic]]:
			edge_set[Vector2i(mini(pair[0], pair[1]), maxi(pair[0], pair[1]))] = true

	var edges: Array = edge_set.keys()
	print("[GPUCloth] Unique edges: %d" % edges.size())

	var vtx_colors: Array = []; vtx_colors.resize(verts.size())
	for i in verts.size(): vtx_colors[i] = {}
	var color_groups: Array = []

	for edge in edges:
		var a: int = edge.x; var b: int = edge.y; var assigned := false
		for c in color_groups.size():
			if (c not in vtx_colors[a]) and (c not in vtx_colors[b]):
				color_groups[c].append(edge)
				vtx_colors[a][c] = true; vtx_colors[b][c] = true
				assigned = true; break
		if not assigned:
			var nc: int = color_groups.size()
			color_groups.append([edge])
			vtx_colors[a][nc] = true; vtx_colors[b][nc] = true

	print("[GPUCloth] Graph coloring: %d groups" % color_groups.size())

	var data := PackedFloat32Array()
	_constraint_groups = []
	for grp in color_groups:
		var start := data.size() / 4
		for edge in grp:
			_push_constraint(data, edge.x, edge.y, verts[edge.x].distance_to(verts[edge.y]), stiffness)
		_constraint_groups.append({offset = start, count = data.size() / 4 - start})

	return data


func _push_constraint(data: PackedFloat32Array, a: int, b: int, rest: float, k: float) -> void:
	data.append(float(a)); data.append(float(b)); data.append(rest); data.append(k)


# ---------------------------------------------------------------------------
#  Collider discovery
# ---------------------------------------------------------------------------

func _find_colliders_recursive(node: Node) -> void:
	if node is GPUClothCollider:
		_colliders.append(node as GPUClothCollider)
	for child in node.get_children():
		_find_colliders_recursive(child)


# ---------------------------------------------------------------------------
#  Bone matrix upload
# ---------------------------------------------------------------------------

func _pack_bone_matrices() -> PackedByteArray:
	var data := PackedByteArray(); data.resize(_bind_count * 48)
	for bi in _bind_count:
		var bone_idx: int = _bind_to_bone[bi]
		if bone_idx < 0:
			continue
		var m: Transform3D = _skeleton_node.get_bone_global_pose(bone_idx) * _skin.get_bind_pose(bi)
		var off := bi * 48
		data.encode_float(off +  0, m.basis.x.x); data.encode_float(off +  4, m.basis.y.x)
		data.encode_float(off +  8, m.basis.z.x); data.encode_float(off + 12, m.origin.x)
		data.encode_float(off + 16, m.basis.x.y); data.encode_float(off + 20, m.basis.y.y)
		data.encode_float(off + 24, m.basis.z.y); data.encode_float(off + 28, m.origin.y)
		data.encode_float(off + 32, m.basis.x.z); data.encode_float(off + 36, m.basis.y.z)
		data.encode_float(off + 40, m.basis.z.z); data.encode_float(off + 44, m.origin.z)
	return data


# ---------------------------------------------------------------------------
#  Collider packing
# ---------------------------------------------------------------------------

func _pack_colliders() -> PackedByteArray:
	if _colliders.is_empty():
		var empty := PackedByteArray(); empty.resize(64); return empty
	var data := PackedByteArray(); data.resize(_colliders.size() * 64)
	var cloth_inv := _skeleton_node.global_transform.affine_inverse()
	for i in _colliders.size():
		var floats := _colliders[i].pack_collider_data(cloth_inv)
		var off := i * 64
		for j in 16: data.encode_float(off + j * 4, floats[j])
	return data


# ---------------------------------------------------------------------------
#  GPU helpers
# ---------------------------------------------------------------------------

func _load_shader(path: String) -> RID:
	var sf: RDShaderFile = load(path)
	if not sf:
		push_error("[GPUCloth] Failed to load shader: %s" % path); return RID()
	var rid := _rd.shader_create_from_spirv(sf.get_spirv())
	if not rid.is_valid():
		push_error("[GPUCloth] Shader compilation failed: %s" % path)
	return rid


func _make_uniform(binding: int, buffer: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buffer)
	return u


func _make_image_uniform(binding: int, img_rid: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u.binding = binding
	u.add_id(img_rid)
	return u


func _create_uniform_set(shader: RID, uniforms: Array[RDUniform]) -> RID:
	return _rd.uniform_set_create(uniforms, shader, 0)
