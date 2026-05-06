@tool
class_name GPUClothSolver
extends Node3D

# ---------------------------------------------------------------------------
#  Exports
# ---------------------------------------------------------------------------
@export_group("Mesh Input")
## MeshInstance3D whose mesh and skeleton will drive the simulation.
## The node will be hidden at runtime; the solver renders its own output mesh.
@export var target_mesh: NodePath
## Skeleton3D that animates the mesh.
@export var skeleton: NodePath
## Which surface index on the ArrayMesh to simulate.
## If the mesh has multiple materials, pick the one that is the cloth (0-based).
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
## Override material for the simulated mesh.
## If left empty the solver copies the material from surface 0 of target_mesh.
@export var cloth_material: Material
## Flip computed normals.  Enable if the cloth appears black/unlit —
## this happens when the mesh winding or skeleton scale inverts the normals.
@export var flip_normals: bool = false

@export_group("Inertia")
@export var inertia_scale: Vector3 = Vector3(1.0, 1.0, 1.0)

@export_group("Wind")
@export var wind: Vector3 = Vector3.ZERO
@export var wind_turbulence: float = 0.3
@export var wind_frequency: float = 1.0

# ---------------------------------------------------------------------------
#  GPU resources
# ---------------------------------------------------------------------------
var _rd: RenderingDevice

# Simulation buffers
var _positions_buffer: RID        # vec4[]: xyz=pos, w=inverse_mass
var _predicted_buffer: RID        # vec4[]: PBD predicted positions
var _velocities_buffer: RID       # vec4[]: velocity
var _constraints_buffer: RID      # vec4[]: (a, b, rest_dist, stiffness)
var _colliders_buffer: RID        # packed collider shapes

# Skinning buffers
var _rest_positions_buffer: RID   # vec4[]: mesh-local rest positions
var _bone_indices_buffer: RID     # uvec2[]: 4 bone indices packed as 2x uint32
var _bone_weights_skin_buffer: RID# vec4[]: 4 blend weights per vertex
var _bone_transforms_buffer: RID  # vec4[]: 3 vec4s per bone (row-major 3x4 mat)
var _skinned_targets_buffer: RID  # vec4[]: output of skin pass (solver-local)
var _cloth_weights_buffer: RID    # vec4[]: x = cloth influence [0..1] per vertex

# Shaders
var _skin_shader: RID
var _predict_shader: RID
var _solve_shader: RID
var _update_shader: RID
var _collide_shader: RID

# Pipelines
var _skin_pipeline: RID
var _predict_pipeline: RID
var _solve_pipeline: RID
var _update_pipeline: RID
var _collide_pipeline: RID

# Uniform sets
var _skin_uniform_set: RID
var _predict_uniform_set: RID
var _solve_uniform_set: RID
var _update_uniform_set: RID
var _collide_uniform_set: RID

# ---------------------------------------------------------------------------
#  Runtime state
# ---------------------------------------------------------------------------
var _mesh_instance_node: MeshInstance3D
var _skeleton_node: Skeleton3D
var _skin: Skin

var _particle_count: int
var _constraint_count: int
var _constraint_groups: Array = []
var _bind_count: int
var _bind_to_bone: PackedInt32Array   # bind_index → skeleton bone index

var _colliders: Array[GPUClothCollider] = []
var _collider_count: int = 0

# Cached mesh topology (built once, reused every frame for normals/tangents)
var _uvs: PackedVector2Array
var _indices: PackedInt32Array
var _orig_tangents: PackedFloat32Array

# Output mesh
var _output_mesh: ArrayMesh
var _output_mesh_instance: MeshInstance3D

# Inertia tracking — we follow the target mesh, not the solver node
var _prev_mesh_world_pos: Vector3

# Reusable per-frame push constant buffers
var _skin_push: PackedByteArray
var _pbd_push: PackedByteArray

var _plugin_dir: String
var _initialized := false
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

	# Run late in the frame so AnimationPlayer (priority 0) has already updated
	# Skeleton3D bone poses before we read them.  Without this, we read last
	# frame's poses and pinned vertices lag one frame behind the animation.
	set_process_priority(100)

	print("[GPUCloth] ── Initializing GPUClothSolver ──────────────────────────")
	_initialize()


func _process(delta: float) -> void:
	if Engine.is_editor_hint() or not _initialized:
		return
	if _needs_warm_start:
		_warm_start()
		_needs_warm_start = false
		return
	_simulate(delta)


func _exit_tree() -> void:
	if not _rd:
		return
	print("[GPUCloth] Freeing GPU resources.")
	for rid in [
		_positions_buffer, _predicted_buffer, _velocities_buffer,
		_constraints_buffer, _colliders_buffer,
		_rest_positions_buffer, _bone_indices_buffer, _bone_weights_skin_buffer,
		_bone_transforms_buffer, _skinned_targets_buffer, _cloth_weights_buffer,
		_skin_pipeline, _predict_pipeline, _solve_pipeline,
		_update_pipeline, _collide_pipeline,
		_skin_shader, _predict_shader, _solve_shader,
		_update_shader, _collide_shader,
	]:
		if rid.is_valid():
			_rd.free_rid(rid)
	_rd.free()


# ---------------------------------------------------------------------------
#  Initialization
# ---------------------------------------------------------------------------

func _initialize() -> void:
	# ── Resolve nodes ───────────────────────────────────────────────────────
	_mesh_instance_node = get_node_or_null(target_mesh) as MeshInstance3D
	if not _mesh_instance_node:
		push_error("[GPUCloth] 'target_mesh' is not set or not a MeshInstance3D.")
		return
	print("[GPUCloth] Target mesh node: %s" % _mesh_instance_node.name)

	_skeleton_node = get_node_or_null(skeleton) as Skeleton3D
	if not _skeleton_node:
		push_error("[GPUCloth] 'skeleton' is not set or not a Skeleton3D.")
		return
	print("[GPUCloth] Skeleton node: %s  (bones: %d)" % [
		_skeleton_node.name, _skeleton_node.get_bone_count()])

	_skin = _mesh_instance_node.get_skin()
	if not _skin:
		push_error("[GPUCloth] target_mesh has no Skin resource. Was it imported from a skinned GLTF/FBX?")
		return
	_bind_count = _skin.get_bind_count()
	print("[GPUCloth] Skin binds: %d" % _bind_count)

	# ── Read ArrayMesh surface ───────────────────────────────────────────────
	var arr_mesh := _mesh_instance_node.mesh as ArrayMesh
	if not arr_mesh:
		push_error("[GPUCloth] target_mesh does not have an ArrayMesh.")
		return
	if arr_mesh.get_surface_count() == 0:
		push_error("[GPUCloth] ArrayMesh has no surfaces.")
		return
	if surface_index >= arr_mesh.get_surface_count():
		push_error("[GPUCloth] surface_index=%d but mesh only has %d surface(s). " \
			% [surface_index, arr_mesh.get_surface_count()] +
			"Set surface_index to the correct cloth surface (0-based).")
		return
	print("[GPUCloth] Mesh has %d surface(s). Simulating surface %d ('%s')." % [
		arr_mesh.get_surface_count(), surface_index,
		arr_mesh.surface_get_name(surface_index)])

	var surf := arr_mesh.surface_get_arrays(surface_index)

	var vert_arr: PackedVector3Array    = surf[Mesh.ARRAY_VERTEX]  if surf[Mesh.ARRAY_VERTEX]  else PackedVector3Array()
	var color_arr: PackedColorArray     = surf[Mesh.ARRAY_COLOR]   if surf[Mesh.ARRAY_COLOR]   else PackedColorArray()
	var bones_raw: PackedInt32Array     = surf[Mesh.ARRAY_BONES]   if surf[Mesh.ARRAY_BONES]   else PackedInt32Array()
	var weights_raw: PackedFloat32Array = surf[Mesh.ARRAY_WEIGHTS] if surf[Mesh.ARRAY_WEIGHTS] else PackedFloat32Array()
	_uvs           = surf[Mesh.ARRAY_TEX_UV]  if surf[Mesh.ARRAY_TEX_UV]  else PackedVector2Array()
	_indices       = surf[Mesh.ARRAY_INDEX]   if surf[Mesh.ARRAY_INDEX]   else PackedInt32Array()
	_orig_tangents = surf[Mesh.ARRAY_TANGENT] if surf[Mesh.ARRAY_TANGENT] else PackedFloat32Array()

	_particle_count = vert_arr.size()
	print("[GPUCloth] Vertices: %d  Triangles: %d" % [_particle_count, _indices.size() / 3])

	if _indices.is_empty():
		push_error("[GPUCloth] Mesh has no index array (ARRAY_INDEX). The mesh must be indexed.")
		return

	# ── Validate required arrays ─────────────────────────────────────────────
	if color_arr.is_empty():
		push_error("[GPUCloth] Mesh has no vertex color data (ARRAY_COLOR). " +
			"In Blender: Object Data → Color Attributes → add one named 'cloth_weight' " +
			"(Domain: Vertex, Type: Byte Color). Paint R=0 for anchored vertices, " +
			"R=1 for simulated. It must be the FIRST color attribute so it exports " +
			"as COLOR_0. Enable 'Vertex Colors' in the GLTF export options.")
		return

	if bones_raw.is_empty():
		push_error("[GPUCloth] Mesh has no bone index data (ARRAY_BONES). " +
			"The mesh must be skinned with a skeleton.")
		return

	if weights_raw.is_empty():
		push_error("[GPUCloth] Mesh has no bone weight data (ARRAY_WEIGHTS).")
		return

	print("[GPUCloth] Bone indices array size: %d  Weights array size: %d" % [
		bones_raw.size(), weights_raw.size()])

	# ── Build cloth weights from vertex color red channel ────────────────────
	var cloth_weights := PackedFloat32Array()
	cloth_weights.resize(_particle_count)
	var n_anchored := 0
	var n_blend    := 0
	var n_free     := 0
	for i in _particle_count:
		var cw: float = clamp(color_arr[i].r, 0.0, 1.0)
		cloth_weights[i] = cw
		if   cw < 0.01:  n_anchored += 1
		elif cw > 0.99:  n_free     += 1
		else:             n_blend    += 1
	print("[GPUCloth] Cloth weight summary → anchored: %d  blend: %d  free: %d" % [
		n_anchored, n_blend, n_free])
	# Print first 8 raw color values so you can verify the vertex colors exported correctly.
	var sample_count := mini(8, _particle_count)
	var sample_str := ""
	for i in sample_count:
		sample_str += "%.2f " % color_arr[i].r
	print("[GPUCloth] First %d raw cloth_weight values (R channel): %s" % [sample_count, sample_str])
	if n_anchored == 0:
		push_warning("[GPUCloth] WARNING: zero anchored vertices! " +
			"The entire cloth will fall under gravity. " +
			"Paint some vertices R=0 (black) on the 'cloth_weight' Color Attribute in Blender.")

	# ── Resolve bind → skeleton bone index ──────────────────────────────────
	_bind_to_bone.resize(_bind_count)
	var unresolved := 0
	for bi in _bind_count:
		var bone_idx: int = _skin.get_bind_bone(bi)
		if bone_idx < 0:
			# Bind uses a name instead of an index
			var bname: String = str(_skin.get_bind_name(bi))
			bone_idx = _skeleton_node.find_bone(bname)
			if bone_idx < 0:
				push_warning("[GPUCloth] Bind %d (name='%s') could not be resolved to a skeleton bone." \
					% [bi, bname])
				unresolved += 1
		_bind_to_bone[bi] = bone_idx
	print("[GPUCloth] Skin bind → bone mapping complete. Unresolved: %d" % unresolved)

	# ── Build position buffers ───────────────────────────────────────────────
	# Working space = Skeleton3D local space.
	# get_bone_global_pose() returns poses in skel-local space, so the skin
	# shader outputs skel-local positions.  The PBD solver must live in the
	# same space.  mesh_to_skel converts mesh-local vertices into skel-local.
	# rest_data stays in mesh-local (it's the input to bind_pose in cloth_skin.glsl).
	var mesh_to_skel := _skeleton_node.global_transform.affine_inverse() \
		* _mesh_instance_node.global_transform
	print("[GPUCloth] mesh_to_skel origin (should be near zero for default rigs): %s" \
		% str(mesh_to_skel.origin))

	var pos_data  := PackedFloat32Array()
	var rest_data := PackedFloat32Array()
	pos_data.resize(_particle_count * 4)
	rest_data.resize(_particle_count * 4)

	for i in _particle_count:
		var skel_pos: Vector3 = mesh_to_skel * vert_arr[i]   # skel-local working space
		var mesh_pos: Vector3 = vert_arr[i]                   # mesh-local for skin shader

		# inverse_mass: 0 = anchored (skeleton pins it), 1 = free PBD
		var inv_mass: float = 0.0 if cloth_weights[i] < 0.01 else 1.0

		pos_data[i * 4 + 0] = skel_pos.x
		pos_data[i * 4 + 1] = skel_pos.y
		pos_data[i * 4 + 2] = skel_pos.z
		pos_data[i * 4 + 3] = inv_mass

		rest_data[i * 4 + 0] = mesh_pos.x
		rest_data[i * 4 + 1] = mesh_pos.y
		rest_data[i * 4 + 2] = mesh_pos.z
		rest_data[i * 4 + 3] = 1.0  # homogeneous

	# ── Build bone index buffer (4 uint16 per vertex → 2 uint32) ────────────
	var bone_idx_bytes := PackedByteArray()
	bone_idx_bytes.resize(_particle_count * 8)
	for i in _particle_count:
		var b: int = i * 4
		var b0: int = bones_raw[b + 0]
		var b1: int = bones_raw[b + 1]
		var b2: int = bones_raw[b + 2]
		var b3: int = bones_raw[b + 3]
		bone_idx_bytes.encode_u32(i * 8 + 0, (b0 & 0xFFFF) | ((b1 & 0xFFFF) << 16))
		bone_idx_bytes.encode_u32(i * 8 + 4, (b2 & 0xFFFF) | ((b3 & 0xFFFF) << 16))
	print("[GPUCloth] Bone index buffer: %d bytes" % bone_idx_bytes.size())

	# ── Build bone weight buffer ─────────────────────────────────────────────
	var bone_w_data := PackedFloat32Array()
	bone_w_data.resize(_particle_count * 4)
	for i in _particle_count:
		var b: int = i * 4
		bone_w_data[b + 0] = weights_raw[b + 0]
		bone_w_data[b + 1] = weights_raw[b + 1]
		bone_w_data[b + 2] = weights_raw[b + 2]
		bone_w_data[b + 3] = weights_raw[b + 3]

	# ── Build cloth weights buffer (vec4, x = influence) ────────────────────
	var cloth_w_data := PackedFloat32Array()
	cloth_w_data.resize(_particle_count * 4)
	for i in _particle_count:
		cloth_w_data[i * 4 + 0] = cloth_weights[i]
		# yzw = 0

	# ── Build velocities buffer (all zero) ──────────────────────────────────
	var vel_data := PackedFloat32Array()
	vel_data.resize(_particle_count * 4)

	# ── Build constraints from mesh topology ─────────────────────────────────
	var con_data := _build_constraints(vert_arr)
	_constraint_count = con_data.size() / 4
	print("[GPUCloth] Constraints: %d total in %d color groups" % [
		_constraint_count, _constraint_groups.size()])

	# ── Discover colliders in skeleton subtree ───────────────────────────────
	_colliders.clear()
	_find_colliders_recursive(_skeleton_node)
	_collider_count = _colliders.size()
	print("[GPUCloth] Colliders found under skeleton: %d" % _collider_count)
	for c in _colliders:
		print("[GPUCloth]   → %s  shape=%s" % [c.get_path(), GPUClothCollider.Shape.keys()[c.shape]])

	# ── Create rendering device ──────────────────────────────────────────────
	_rd = RenderingServer.create_local_rendering_device()
	print("[GPUCloth] Local RenderingDevice created.")

	# ── Upload buffers ───────────────────────────────────────────────────────
	var pos_bytes      := pos_data.to_byte_array()
	var rest_bytes     := rest_data.to_byte_array()
	var vel_bytes      := vel_data.to_byte_array()
	var con_bytes      := con_data.to_byte_array()
	var bone_w_bytes   := bone_w_data.to_byte_array()
	var cloth_w_bytes  := cloth_w_data.to_byte_array()
	var bone_mat_bytes := _pack_bone_matrices()
	var col_bytes      := _pack_colliders()

	_positions_buffer         = _rd.storage_buffer_create(pos_bytes.size(),      pos_bytes)
	_predicted_buffer         = _rd.storage_buffer_create(pos_bytes.size(),      pos_bytes)
	_velocities_buffer        = _rd.storage_buffer_create(vel_bytes.size(),      vel_bytes)
	_constraints_buffer       = _rd.storage_buffer_create(max(con_bytes.size(), 64), con_bytes)
	_colliders_buffer         = _rd.storage_buffer_create(max(col_bytes.size(), 64), col_bytes)
	_rest_positions_buffer    = _rd.storage_buffer_create(rest_bytes.size(),     rest_bytes)
	_bone_indices_buffer      = _rd.storage_buffer_create(bone_idx_bytes.size(), bone_idx_bytes)
	_bone_weights_skin_buffer = _rd.storage_buffer_create(bone_w_bytes.size(),   bone_w_bytes)
	_bone_transforms_buffer   = _rd.storage_buffer_create(max(bone_mat_bytes.size(), 64), bone_mat_bytes)
	_cloth_weights_buffer     = _rd.storage_buffer_create(cloth_w_bytes.size(),  cloth_w_bytes)
	# skinned_targets: same size as positions, initialized to rest positions
	_skinned_targets_buffer   = _rd.storage_buffer_create(pos_bytes.size(),      pos_bytes)

	print("[GPUCloth] All GPU buffers uploaded.")
	print("[GPUCloth]   positions:      %d bytes" % pos_bytes.size())
	print("[GPUCloth]   rest_positions: %d bytes" % rest_bytes.size())
	print("[GPUCloth]   bone_indices:   %d bytes" % bone_idx_bytes.size())
	print("[GPUCloth]   bone_weights:   %d bytes" % bone_w_bytes.size())
	print("[GPUCloth]   bone_transforms:%d bytes (%d bones)" % [bone_mat_bytes.size(), _bind_count])
	print("[GPUCloth]   constraints:    %d bytes" % con_bytes.size())

	# ── Load shaders ─────────────────────────────────────────────────────────
	_skin_shader    = _load_shader(_plugin_dir + "/shaders/compute/cloth_skin.glsl")
	_predict_shader = _load_shader(_plugin_dir + "/shaders/compute/cloth_predict.glsl")
	_solve_shader   = _load_shader(_plugin_dir + "/shaders/compute/cloth_solve.glsl")
	_update_shader  = _load_shader(_plugin_dir + "/shaders/compute/cloth_update.glsl")
	_collide_shader = _load_shader(_plugin_dir + "/shaders/compute/cloth_collide.glsl")
	print("[GPUCloth] Shaders compiled.")

	# ── Create pipelines ─────────────────────────────────────────────────────
	_skin_pipeline    = _rd.compute_pipeline_create(_skin_shader)
	_predict_pipeline = _rd.compute_pipeline_create(_predict_shader)
	_solve_pipeline   = _rd.compute_pipeline_create(_solve_shader)
	_update_pipeline  = _rd.compute_pipeline_create(_update_shader)
	_collide_pipeline = _rd.compute_pipeline_create(_collide_shader)
	print("[GPUCloth] Compute pipelines created.")

	# ── Create uniform sets ───────────────────────────────────────────────────
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
	print("[GPUCloth] Uniform sets created.")

	# ── Build reusable push constant buffers ─────────────────────────────────
	_skin_push = PackedByteArray()
	_skin_push.resize(64)
	# particle_count at byte 0, bone_count at byte 4
	_skin_push.encode_u32(0, _particle_count)
	_skin_push.encode_u32(4, _bind_count)

	_pbd_push = PackedByteArray()
	_pbd_push.resize(64)

	# ── Output mesh instance ─────────────────────────────────────────────────
	# Add as a sibling of the target mesh (child of same parent) at the same
	# local transform, so it renders in the same coordinate space as the original.
	_output_mesh = ArrayMesh.new()
	_output_mesh_instance = MeshInstance3D.new()
	_output_mesh_instance.mesh = _output_mesh
	_output_mesh_instance.name = "GPUClothOutput"
	if cloth_material:
		_output_mesh_instance.material_override = cloth_material
	else:
		var mat := _mesh_instance_node.get_active_material(surface_index)
		if mat:
			_output_mesh_instance.material_override = mat
			print("[GPUCloth] Using material from surface %d: %s" % [surface_index, mat.resource_name])
		else:
			push_warning("[GPUCloth] No material found on surface %d; mesh may appear white." % surface_index)

	# The solver works in Skeleton3D local space, so the output mesh MUST be a
	# direct child of the Skeleton3D with an identity transform.  Do NOT add it
	# to the mesh's parent (e.g. an Armature Node3D above the skeleton) — that
	# would introduce an extra offset equal to the skeleton's position within
	# its own parent.
	_skeleton_node.add_child(_output_mesh_instance)
	_output_mesh_instance.transform = Transform3D.IDENTITY
	print("[GPUCloth] Output mesh added as child of Skeleton3D '%s' (identity transform, skel-local verts)." \
		% _skeleton_node.name)

	# Hide only the simulated surface on the original mesh by replacing it with a
	# fully-transparent material override.  All other surfaces remain visible so
	# the body/armour/etc. still render via Godot's native skinning.
	var invisible_mat := StandardMaterial3D.new()
	invisible_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	invisible_mat.albedo_color  = Color(0.0, 0.0, 0.0, 0.0)
	invisible_mat.cull_mode     = BaseMaterial3D.CULL_DISABLED
	_mesh_instance_node.set_surface_override_material(surface_index, invisible_mat)
	print("[GPUCloth] Surface %d on original mesh hidden via transparent override." % surface_index)

	# Log a few initial positions so you can sanity-check the working space.
	# These should be close to the skeleton-local coordinates of cloth verts
	# (typically small values, e.g. near body origin).
	var sample_pos_count := mini(4, _particle_count)
	for i in sample_pos_count:
		var px := pos_data[i * 4 + 0]; var py := pos_data[i * 4 + 1]; var pz := pos_data[i * 4 + 2]
		var iw := pos_data[i * 4 + 3]
		print("[GPUCloth] init pos[%d] = (%.3f, %.3f, %.3f)  inv_mass=%.0f" % [i, px, py, pz, iw])

	# Build the initial display mesh from rest positions.
	_update_mesh(pos_bytes)

	_prev_mesh_world_pos = _skeleton_node.global_position
	_initialized = true

	print("[GPUCloth] ── Initialization complete ──────────────────────────────")
	print("[GPUCloth]   particles: %d  constraints: %d  groups: %d  colliders: %d" % [
		_particle_count, _constraint_count, _constraint_groups.size(), _collider_count])


# ---------------------------------------------------------------------------
#  Warm start — runs on the first _process tick (after AnimationPlayer)
# ---------------------------------------------------------------------------

func _warm_start() -> void:
	# Upload bone matrices that reflect the current animation pose.
	var bone_bytes := _pack_bone_matrices()
	_rd.buffer_update(_bone_transforms_buffer, 0, bone_bytes.size(), bone_bytes)

	# Run the skin pass once to populate skinned_targets with animated positions.
	var groups := ceili(float(_particle_count) / 64.0)
	var cl := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(cl, _skin_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _skin_uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, _skin_push, 64)
	_rd.compute_list_dispatch(cl, groups, 1, 1)
	_rd.compute_list_end()
	_rd.submit()
	_rd.sync()

	# Read back the skinned positions and teleport ALL particles there so
	# simulation starts from the actual animated pose, not the mesh rest pose.
	var skinned  := _rd.buffer_get_data(_skinned_targets_buffer)
	var cur_pos  := _rd.buffer_get_data(_positions_buffer)
	var new_pos  := PackedByteArray()
	new_pos.resize(_particle_count * 16)
	for i in _particle_count:
		var off := i * 16
		new_pos.encode_float(off +  0, skinned.decode_float(off + 0))
		new_pos.encode_float(off +  4, skinned.decode_float(off + 4))
		new_pos.encode_float(off +  8, skinned.decode_float(off + 8))
		new_pos.encode_float(off + 12, cur_pos.decode_float(off + 12))  # preserve inv_mass
	_rd.buffer_update(_positions_buffer, 0, new_pos.size(), new_pos)
	_rd.buffer_update(_predicted_buffer, 0, new_pos.size(), new_pos)

	# Zero velocities so no initial impulse.
	var zero_vel := PackedByteArray(); zero_vel.resize(_particle_count * 16)
	_rd.buffer_update(_velocities_buffer, 0, zero_vel.size(), zero_vel)

	# Re-anchor inertia tracking to avoid a spurious kick on the first simulate.
	_prev_mesh_world_pos = _skeleton_node.global_position

	_update_mesh(new_pos)
	print("[GPUCloth] Warm start complete — particles set to current animation pose.")


# ---------------------------------------------------------------------------
#  Per-frame simulation
# ---------------------------------------------------------------------------

func _simulate(delta: float) -> void:
	var sub_dt := delta / float(substeps)

	# ── Upload updated bone matrices (cheap: ~64–256 matrix reads from CPU) ──
	var bone_bytes := _pack_bone_matrices()
	_rd.buffer_update(_bone_transforms_buffer, 0, bone_bytes.size(), bone_bytes)

	# ── Upload updated collider transforms ───────────────────────────────────
	if _collider_count > 0:
		var cb := _pack_colliders()
		_rd.buffer_update(_colliders_buffer, 0, cb.size(), cb)

	# ── Inertia: compensate for the skeleton moving in world space ────────
	# Solver works in skel-local space, so we track the skeleton's world position.
	var delta_world := _skeleton_node.global_position - _prev_mesh_world_pos
	var delta_local := _skeleton_node.global_transform.basis.inverse() * delta_world
	var inertia_sub := delta_local * inertia_scale / float(substeps)
	_prev_mesh_world_pos = _skeleton_node.global_position

	# ── Wind ─────────────────────────────────────────────────────────────────
	var t := Time.get_ticks_msec() / 1000.0 * wind_frequency
	var gust := Vector3(
		sin(t * 1.7) + sin(t * 3.1 + 1.3),
		sin(t * 1.3 + 2.0) + sin(t * 2.7 + 0.7),
		sin(t * 2.1 + 4.0) + sin(t * 1.9 + 3.1)
	) * 0.5
	var eff_wind := wind + wind.length() * gust * wind_turbulence
	var local_wind := global_transform.basis.inverse() * eff_wind

	# ── Pack PBD push constants ───────────────────────────────────────────────
	_pbd_push.encode_float(0,  sub_dt)
	_pbd_push.encode_float(4,  gravity_strength)
	_pbd_push.encode_u32(8,    _particle_count)
	_pbd_push.encode_u32(12,   _constraint_count)  # overwritten per-group in solve
	_pbd_push.encode_float(16, damping)
	_pbd_push.encode_float(20, max_speed)
	_pbd_push.encode_u32(24,   _collider_count)
	_pbd_push.encode_u32(28,   0)                  # constraint_offset placeholder
	_pbd_push.encode_float(32, inertia_sub.x)
	_pbd_push.encode_float(36, inertia_sub.y)
	_pbd_push.encode_float(40, inertia_sub.z)
	_pbd_push.encode_float(44, max_travel_distance)
	_pbd_push.encode_float(48, local_wind.x)
	_pbd_push.encode_float(52, local_wind.y)
	_pbd_push.encode_float(56, local_wind.z)
	_pbd_push.encode_float(60, 0.0)

	var groups := ceili(float(_particle_count) / 64.0)

	var cl := _rd.compute_list_begin()

	# ── SKIN PASS (once per frame, before substeps) ───────────────────────────
	# Reads bone matrices + rest positions → writes skinned_targets.
	# Anchored particles in predict will read from skinned_targets.
	_rd.compute_list_bind_compute_pipeline(cl, _skin_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _skin_uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, _skin_push, 64)
	_rd.compute_list_dispatch(cl, groups, 1, 1)
	_rd.compute_list_add_barrier(cl)

	for _s in substeps:
		# PREDICT: apply forces, anchor particles to skinned_targets
		_rd.compute_list_bind_compute_pipeline(cl, _predict_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _predict_uniform_set, 0)
		_rd.compute_list_set_push_constant(cl, _pbd_push, 64)
		_rd.compute_list_dispatch(cl, groups, 1, 1)
		_rd.compute_list_add_barrier(cl)

		# SOLVE + COLLIDE interleaved: collision constraints participate in each
		# iteration so distance constraints cannot repeatedly pull particles back
		# through a collider surface between collision responses.
		_rd.compute_list_bind_compute_pipeline(cl, _solve_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _solve_uniform_set, 0)
		for _iter in solver_iterations:
			for grp in _constraint_groups:
				_pbd_push.encode_u32(12, grp.count)
				_pbd_push.encode_u32(28, grp.offset)
				_rd.compute_list_set_push_constant(cl, _pbd_push, 64)
				_rd.compute_list_dispatch(cl, ceili(float(grp.count) / 64.0), 1, 1)
				_rd.compute_list_add_barrier(cl)

			if _collider_count > 0:
				_pbd_push.encode_u32(12, _constraint_count)
				_pbd_push.encode_u32(28, 0)
				_rd.compute_list_bind_compute_pipeline(cl, _collide_pipeline)
				_rd.compute_list_bind_uniform_set(cl, _collide_uniform_set, 0)
				_rd.compute_list_set_push_constant(cl, _pbd_push, 64)
				_rd.compute_list_dispatch(cl, groups, 1, 1)
				_rd.compute_list_add_barrier(cl)
				_rd.compute_list_bind_compute_pipeline(cl, _solve_pipeline)
				_rd.compute_list_bind_uniform_set(cl, _solve_uniform_set, 0)

		# UPDATE: recover velocity, blend toward skinned_targets in blend zone
		_rd.compute_list_bind_compute_pipeline(cl, _update_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _update_uniform_set, 0)
		_rd.compute_list_set_push_constant(cl, _pbd_push, 64)
		_rd.compute_list_dispatch(cl, groups, 1, 1)
		_rd.compute_list_add_barrier(cl)

	_rd.compute_list_end()
	_rd.submit()
	_rd.sync()

	var out_bytes := _rd.buffer_get_data(_positions_buffer)
	_update_mesh(out_bytes)


# ---------------------------------------------------------------------------
#  Constraint building — edge extraction + greedy graph (edge) coloring
# ---------------------------------------------------------------------------

func _build_constraints(verts: PackedVector3Array) -> PackedFloat32Array:
	print("[GPUCloth] Building constraints from %d triangles..." % (_indices.size() / 3))

	# ── Step 1: extract unique edges ─────────────────────────────────────────
	var edge_set: Dictionary = {}
	var tri_count := _indices.size() / 3
	for t in tri_count:
		var ia := _indices[t * 3 + 0]
		var ib := _indices[t * 3 + 1]
		var ic := _indices[t * 3 + 2]
		for pair in [[ia, ib], [ib, ic], [ia, ic]]:
			var lo := mini(pair[0], pair[1])
			var hi := maxi(pair[0], pair[1])
			edge_set[Vector2i(lo, hi)] = true

	var edges: Array = edge_set.keys()
	print("[GPUCloth] Unique edges: %d" % edges.size())

	# ── Step 2: greedy edge coloring ─────────────────────────────────────────
	# Each vertex tracks which color groups it already participates in.
	# An edge is assigned the first color where neither endpoint is already used.
	# This ensures no two edges in a group share a vertex → no GPU write conflicts.
	var vtx_colors: Array = []  # vtx_idx → Dictionary{color_idx: true}
	vtx_colors.resize(verts.size())
	for i in verts.size():
		vtx_colors[i] = {}

	var color_groups: Array = []  # Array of Array[Vector2i]

	for edge in edges:
		var a: int = edge.x
		var b: int = edge.y
		var assigned := false
		for c in color_groups.size():
			if (c not in vtx_colors[a]) and (c not in vtx_colors[b]):
				color_groups[c].append(edge)
				vtx_colors[a][c] = true
				vtx_colors[b][c] = true
				assigned = true
				break
		if not assigned:
			var nc: int = color_groups.size()
			color_groups.append([edge])
			vtx_colors[a][nc] = true
			vtx_colors[b][nc] = true

	print("[GPUCloth] Graph coloring complete: %d groups (max parallel dispatch groups)" \
		% color_groups.size())

	# ── Step 3: pack into flat constraint buffer ──────────────────────────────
	var data := PackedFloat32Array()
	_constraint_groups = []
	for grp in color_groups:
		var start := data.size() / 4
		for edge in grp:
			var a: int = edge.x
			var b: int = edge.y
			var dist: float = verts[a].distance_to(verts[b])
			_push_constraint(data, a, b, dist, stiffness)
		_constraint_groups.append({offset = start, count = data.size() / 4 - start})

	return data


func _push_constraint(data: PackedFloat32Array, a: int, b: int, rest: float, k: float) -> void:
	data.append(float(a))
	data.append(float(b))
	data.append(rest)
	data.append(k)


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
	# Each bind slot produces one 3×4 matrix (12 floats = 48 bytes).
	# get_bone_global_pose() returns the bone transform in Skeleton3D local space.
	# bind_pose (from skin.get_bind_pose) transforms mesh-local → bone-local.
	# Combined: bone_pose * bind_pose maps mesh-local rest positions → skel-local.
	# This matches the working space of the PBD solver (skel-local).
	var data := PackedByteArray()
	data.resize(_bind_count * 48)

	for bi in _bind_count:
		var bone_idx: int = _bind_to_bone[bi]
		if bone_idx < 0:
			# Unresolved bind: write zero matrix (no contribution)
			continue

		var bone_pose: Transform3D = _skeleton_node.get_bone_global_pose(bone_idx)
		var bind_pose: Transform3D = _skin.get_bind_pose(bi)
		var m: Transform3D         = bone_pose * bind_pose

		# Pack as row-major 3×4:
		#   row0 = (m.basis.x.x, m.basis.y.x, m.basis.z.x, m.origin.x)
		#   row1 = (m.basis.x.y, m.basis.y.y, m.basis.z.y, m.origin.y)
		#   row2 = (m.basis.x.z, m.basis.y.z, m.basis.z.z, m.origin.z)
		var off := bi * 48
		data.encode_float(off +  0, m.basis.x.x); data.encode_float(off +  4, m.basis.y.x)
		data.encode_float(off +  8, m.basis.z.x); data.encode_float(off + 12, m.origin.x)
		data.encode_float(off + 16, m.basis.x.y); data.encode_float(off + 20, m.basis.y.y)
		data.encode_float(off + 24, m.basis.z.y); data.encode_float(off + 28, m.origin.y)
		data.encode_float(off + 32, m.basis.x.z); data.encode_float(off + 36, m.basis.y.z)
		data.encode_float(off + 40, m.basis.z.z); data.encode_float(off + 44, m.origin.z)

	return data


# ---------------------------------------------------------------------------
#  Collider packing (unchanged from original, now scanned from skeleton)
# ---------------------------------------------------------------------------

func _pack_colliders() -> PackedByteArray:
	if _colliders.is_empty():
		var empty := PackedByteArray()
		empty.resize(64)
		return empty
	var data := PackedByteArray()
	data.resize(_colliders.size() * 64)
	# Particles live in Skeleton3D local space, so colliders must be transformed
	# into that same space — not into the solver node's local space.
	var cloth_inv := _skeleton_node.global_transform.affine_inverse()
	for i in _colliders.size():
		var floats := _colliders[i].pack_collider_data(cloth_inv)
		var off := i * 64
		for j in 16:
			data.encode_float(off + j * 4, floats[j])
	return data


# ---------------------------------------------------------------------------
#  Mesh output — rebuild ArrayMesh each frame from GPU position readback
# ---------------------------------------------------------------------------

func _update_mesh(data: PackedByteArray) -> void:
	var verts := PackedVector3Array()
	verts.resize(_particle_count)
	for i in _particle_count:
		var off := i * 16
		verts[i] = Vector3(
			data.decode_float(off),
			data.decode_float(off + 4),
			data.decode_float(off + 8))

	# ── Normals: per-face cross products accumulated per vertex ───────────────
	var normals := PackedVector3Array()
	normals.resize(_particle_count)

	var tri_count := _indices.size() / 3
	for t in tri_count:
		var i0 := _indices[t * 3 + 0]
		var i1 := _indices[t * 3 + 1]
		var i2 := _indices[t * 3 + 2]
		var v0  := verts[i0]
		var n   := (verts[i1] - v0).cross(verts[i2] - v0)
		normals[i0] += n
		normals[i1] += n
		normals[i2] += n

	var nflip := -1.0 if flip_normals else 1.0
	for i in _particle_count:
		var nl := normals[i].length_squared()
		normals[i] = normals[i] * (nflip / sqrt(nl)) if nl > 1e-8 else Vector3.UP * nflip

	# ── Tangents: from UV derivatives, or fallback to stored originals ────────
	var tangents := _compute_tangents(verts, normals)

	var arrays: Array = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX]  = verts
	arrays[Mesh.ARRAY_NORMAL]  = normals
	arrays[Mesh.ARRAY_TANGENT] = tangents
	arrays[Mesh.ARRAY_INDEX]   = _indices
	if not _uvs.is_empty():
		arrays[Mesh.ARRAY_TEX_UV] = _uvs

	_output_mesh.clear_surfaces()
	_output_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)


func _compute_tangents(verts: PackedVector3Array, normals: PackedVector3Array) -> PackedFloat32Array:
	var tangents := PackedFloat32Array()
	tangents.resize(_particle_count * 4)

	if _uvs.is_empty():
		if not _orig_tangents.is_empty():
			return _orig_tangents
		for i in _particle_count:
			tangents[i * 4 + 0] = 1.0
			tangents[i * 4 + 3] = 1.0
		return tangents

	# Accumulate tangent (T) and bitangent (B) per vertex from UV derivatives.
	var tan_accum  := PackedVector3Array(); tan_accum.resize(_particle_count)
	var btan_accum := PackedVector3Array(); btan_accum.resize(_particle_count)

	var tri_count := _indices.size() / 3
	for t in tri_count:
		var i0 := _indices[t * 3 + 0]
		var i1 := _indices[t * 3 + 1]
		var i2 := _indices[t * 3 + 2]
		var e1   := verts[i1] - verts[i0]
		var e2   := verts[i2] - verts[i0]
		var duv1 := _uvs[i1]  - _uvs[i0]
		var duv2 := _uvs[i2]  - _uvs[i0]
		var denom := duv1.x * duv2.y - duv2.x * duv1.y
		if abs(denom) < 1e-10:
			continue
		var f    := 1.0 / denom
		var tang := (e1 * duv2.y - e2 * duv1.y) * f
		var btan := (e2 * duv1.x - e1 * duv2.x) * f
		tan_accum[i0]  += tang; tan_accum[i1]  += tang; tan_accum[i2]  += tang
		btan_accum[i0] += btan; btan_accum[i1] += btan; btan_accum[i2] += btan

	for i in _particle_count:
		var tl := tan_accum[i].length_squared()
		var tv := tan_accum[i] / sqrt(tl) if tl > 1e-10 else Vector3.RIGHT
		# Bitangent sign: Godot computes binormal = cross(normal, tangent) * w.
		# w must be +1 or -1 so that binormal matches the UV-derived bitangent.
		var w := 1.0 if normals[i].cross(tv).dot(btan_accum[i]) >= 0.0 else -1.0
		tangents[i * 4 + 0] = tv.x
		tangents[i * 4 + 1] = tv.y
		tangents[i * 4 + 2] = tv.z
		tangents[i * 4 + 3] = w

	return tangents


# ---------------------------------------------------------------------------
#  GPU helpers
# ---------------------------------------------------------------------------

func _load_shader(path: String) -> RID:
	print("[GPUCloth] Loading shader: %s" % path)
	var sf: RDShaderFile = load(path)
	if not sf:
		push_error("[GPUCloth] Failed to load shader file: %s" % path)
		return RID()
	var spirv := sf.get_spirv()
	var rid := _rd.shader_create_from_spirv(spirv)
	if not rid.is_valid():
		push_error("[GPUCloth] Shader compilation failed: %s" % path)
	return rid


func _make_uniform(binding: int, buffer: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buffer)
	return u


func _create_uniform_set(shader: RID, uniforms: Array[RDUniform]) -> RID:
	return _rd.uniform_set_create(uniforms, shader, 0)
