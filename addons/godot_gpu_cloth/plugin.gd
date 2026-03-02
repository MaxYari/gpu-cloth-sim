@tool
extends EditorPlugin


func _enter_tree() -> void:
	add_custom_type(
		"GPUClothSolver",
		"Node3D",
		preload("src/gpu_cloth_solver.gd"),
		preload("icons/gpu_cloth_solver.svg")
	)
	add_custom_type(
		"GPUClothCollider",
		"Node3D",
		preload("src/gpu_cloth_collider.gd"),
		preload("icons/gpu_cloth_collider.svg")
	)


func _exit_tree() -> void:
	remove_custom_type("GPUClothSolver")
	remove_custom_type("GPUClothCollider")
