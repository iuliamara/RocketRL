extends Node3D

var hud := Label.new()


func _ready() -> void:
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-52, -28, 0)
	sun.light_energy = 1.15
	sun.shadow_enabled = true
	add_child(sun)
	var environment_node := WorldEnvironment.new()
	environment_node.environment = Environment.new()
	environment_node.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment_node.environment.ambient_light_color = Color(0.67, 0.78, 0.93)
	environment_node.environment.ambient_light_energy = 0.62
	add_child(environment_node)
	var camera := Camera3D.new()
	add_child(camera)
	camera.position = Vector3(13, 10, 17)
	camera.look_at(Vector3(0, 3.1, 0))
	camera.fov = 49
	add_box(self, Vector3(32, 0.2, 32), Vector3(0, -0.105, 0), Color("#293c50"))
	add_box(self, Vector3(3.3, 0.012, 3.3), Vector3.ZERO, Color("#306454"))
	for side in [-1, 1]:
		add_box(self, Vector3(3.3, 0.015, 0.065), Vector3(0, 0.01, side * 1.62), Color("#b6f5bb"))
		add_box(self, Vector3(0.065, 0.015, 3.3), Vector3(side * 1.62, 0.01, 0), Color("#b6f5bb"))
	for line in range(-14, 15, 2):
		if absi(line) >= 2:
			add_box(self, Vector3(28, 0.005, 0.022), Vector3(0, -0.002, line), Color("#42556b"))
			add_box(self, Vector3(0.022, 0.005, 28), Vector3(line, -0.002, 0), Color("#42556b"))
	var overlay := CanvasLayer.new()
	add_child(overlay)
	overlay.add_child(hud)
	hud.position = Vector2(26, 23)
	hud.add_theme_font_size_override("font_size", 20)
	hud.add_theme_color_override("font_shadow_color", Color.BLACK)
	hud.add_theme_constant_override("shadow_offset_x", 1)
	hud.add_theme_constant_override("shadow_offset_y", 2)


func build_rocket(rocket: Node3D) -> void:
	var shell := CylinderMesh.new()
	shell.height = 1.36
	shell.top_radius = 0.22
	shell.bottom_radius = 0.30
	mesh(rocket, shell, Vector3(0, -0.12, 0), Color("#e1e9f2"))
	var nose := CylinderMesh.new()
	nose.height = 0.24
	nose.top_radius = 0.04
	nose.bottom_radius = 0.22
	mesh(rocket, nose, Vector3(0, 0.68, 0), Color("#ef9161"))
	add_box(rocket, Vector3(0.13, 0.1, 0.13), Vector3(0, -0.76, 0), Color("#ef9161"))


func add_box(parent: Node3D, size: Vector3, position: Vector3, color: Color) -> void:
	var box := BoxMesh.new()
	box.size = size
	mesh(parent, box, position, color)


func mesh(parent: Node3D, geometry: Mesh, position: Vector3, color: Color) -> void:
	var item := MeshInstance3D.new()
	item.mesh = geometry
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	item.material_override = material
	item.position = position
	parent.add_child(item)


func update_hud(rocket: RigidBody3D, fuel: float, reason: String, manual: bool) -> void:
	var state := reason.replace("_", " ").capitalize() if not reason.is_empty() else "Descending"
	hud.text = "RocketRL 3D   |   %s\nHeight %.2f   ·   Speed %.2f   ·   Fuel %.2f" % [
		state, rocket.position.y, rocket.linear_velocity.length(), fuel]
	if manual:
		hud.text += "\nSpace / Shift  throttle   ·   WASD  gimbal   ·   R  reset"
