extends Node3D

const Rocket = preload("res://rocket.gd")
const Rules = preload("res://landing_rules.gd")
const Bridge = preload("res://bridge.gd")
const Visuals = preload("res://visuals.gd")
const ACTION_TICKS := 4
const IDLE := [-1.0, 0.0, 0.0]

var rocket: Rocket
var visuals: Visuals
var bridge: Bridge
var rng := RandomNumberGenerator.new()
var wind := Vector3.ZERO
var ticks := 0
var remaining := 0
var previous_cost := 0.0
var previous_velocity := Vector3.ZERO
var command := IDLE
var reason := ""
var step_burn := 0.0
var terminal_velocity := Vector3.ZERO
var manual := true
var port := 0
var reset_was_down := false


func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--rl-port="):
			port = int(arg.get_slice("=", 1))
	manual = port == 0
	var ground := StaticBody3D.new()
	var collider := CollisionShape3D.new()
	var plane := WorldBoundaryShape3D.new()
	collider.shape = plane
	ground.add_child(collider)
	add_child(ground)
	rocket = Rocket.new()
	add_child(rocket)
	if DisplayServer.get_name() != "headless":
		visuals = Visuals.new()
		add_child(visuals)
		visuals.build_rocket(rocket)
	reset_episode({"seed": 11})
	if not manual:
		bridge = Bridge.new()
		if not bridge.connect_local(port):
			push_error("Could not connect to the RocketRL Python process.")
			get_tree().quit(1)


func reset_episode(message: Dictionary) -> void:
	rng.seed = int(message.get("seed", 0))
	var conditions: Dictionary = message.get("conditions", {})
	var position := Vector3(
		rng.randf_range(-2.2, 2.2) * float(conditions.get("offset_scale", 1.0)),
		rng.randf_range(3.8, 6.2) + float(conditions.get("altitude_offset", 0.0)),
		rng.randf_range(-2.2, 2.2) * float(conditions.get("offset_scale", 1.0))
	)
	var speed := float(conditions.get("speed_scale", 1.0))
	var velocity := Vector3(rng.randf_range(-0.55, 0.55), rng.randf_range(-1.15, -0.35),
		rng.randf_range(-0.55, 0.55)) * speed
	var tilt := Vector2(rng.randf_range(-0.09, 0.09), rng.randf_range(-0.09, 0.09))
	var gust := float(conditions.get("wind", 0.0))
	var wind_angle := rng.randf_range(0, TAU)
	wind = Vector3(cos(wind_angle), 0, sin(wind_angle)) * gust
	rocket.freeze = false
	rocket.reset_body(position, velocity, tilt, float(conditions.get("fuel_fraction", 1.0)),
		float(conditions.get("payload", 0.0)), float(conditions.get("efficiency", 1.0)))
	ticks = 0
	remaining = 0
	reason = ""
	step_burn = 0.0
	previous_velocity = velocity
	terminal_velocity = Vector3.ZERO


func observation() -> Dictionary:
	var p := rocket.position
	var v := rocket.linear_velocity
	var up := rocket.global_basis.y
	var omega := rocket.angular_velocity
	var values := [
		p.x / 12.0, (p.y - Rocket.HALF_HEIGHT) / 12.0, p.z / 12.0,
		v.x / 8.0, v.y / 8.0, v.z / 8.0, up.x, up.y, up.z,
		omega.x / 4.0, omega.y / 4.0, omega.z / 4.0,
		rocket.fuel / Rocket.FUEL_CAPACITY, rocket.payload / 0.6,
		rocket.efficiency - 1.0, wind.x, wind.z, float(ticks) / Rules.MAX_TICKS,
	]
	for index in range(values.size()):
		values[index] = clampf(values[index], -1.0, 1.0)
	return {"state": values}


func measure() -> void:
	if ticks == 0 or not reason.is_empty():
		return
	reason = Rules.outcome(rocket.position, rocket.global_basis.y, previous_velocity,
		rocket.angular_velocity, not rocket.get_colliding_bodies().is_empty(), ticks)
	if not reason.is_empty():
		terminal_velocity = previous_velocity


func _physics_process(delta: float) -> void:
	measure()
	if visuals:
		visuals.update_hud(rocket, rocket.fuel, reason, manual)
	if manual:
		var reset_down := Input.is_physical_key_pressed(KEY_R)
		if reset_down and not reset_was_down:
			reset_episode({"seed": rng.randi()})
		reset_was_down = reset_down
		if reason.is_empty():
			advance(manual_action(), delta)
		else:
			rocket.freeze = true
		return
	if remaining > 0 and reason.is_empty():
		remaining -= 1
		advance(command, delta)
		return
	if ticks > 0:
		if not send_step():
			get_tree().quit()
			return
	# Wait in this physics callback so one policy step always spans exactly four ticks
	# unless an episode ends on an earlier tick.
	while true:
		var message := bridge.receive()
		match message.get("type", ""):
			"handshake":
				if str(message.get("major_version")) != "0" or str(message.get("minor_version")) != "7":
					get_tree().quit(1)
					return
			"env_info":
				bridge.send({
					"type": "env_info", "n_agents": 1,
					"action_space": {"control": {"action_type": "continuous", "size": 3}},
					"observation_space": {"state": {"space": "box", "size": [18]}},
				})
			"reset":
				reset_episode(message)
				bridge.send({"type": "reset", "obs": [observation()]})
			"action":
				if not reason.is_empty():
					get_tree().quit(1)
					return
				command = message["action"][0]["control"]
				previous_cost = Rules.state_cost(rocket.position, rocket.linear_velocity, rocket.global_basis.y)
				step_burn = rocket.fuel
				remaining = ACTION_TICKS - 1
				advance(command, delta)
				return
			_:
				get_tree().quit()
				return


func advance(action: Array, delta: float) -> void:
	previous_velocity = rocket.linear_velocity
	rocket.thrust(action, wind, delta)
	ticks += 1


func send_step() -> bool:
	var done := not reason.is_empty()
	var reward := previous_cost - Rules.state_cost(rocket.position, rocket.linear_velocity, rocket.global_basis.y)
	reward += Rules.terminal_reward(reason) - 0.025 - 2.0 * (step_burn - rocket.fuel)
	var info := {
		"reason": reason, "success": reason == "landing", "ticks": ticks,
		"center_error": Vector2(rocket.position.x, rocket.position.z).length(),
		"fuel_used": rocket.loaded_fuel - rocket.fuel,
		"up_dot": rocket.global_basis.y.y,
		"impact_speed": terminal_velocity.length() if reason in ["landing", "off_pad", "hard_contact"] else null,
	}
	return bridge.send({"type": "step", "obs": [observation()], "reward": [reward], "done": [done], "info": [info]})


func manual_action() -> Array:
	var throttle := 0.0
	if Input.is_physical_key_pressed(KEY_SPACE):
		throttle = 1.0
	elif Input.is_physical_key_pressed(KEY_SHIFT):
		throttle = 0.44
	var x := int(Input.is_physical_key_pressed(KEY_D)) - int(Input.is_physical_key_pressed(KEY_A))
	var z := int(Input.is_physical_key_pressed(KEY_S)) - int(Input.is_physical_key_pressed(KEY_W))
	return [throttle * 2.0 - 1.0, float(x), float(z)]
