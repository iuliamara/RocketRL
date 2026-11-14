extends RigidBody3D

const DRY_MASS := 1.0
const FUEL_CAPACITY := 0.40
const BURN_RATE := 0.055
const THRUST := 10.0
const GIMBAL := 0.38
const HALF_HEIGHT := 0.80

var fuel: float
var loaded_fuel: float
var payload: float
var efficiency: float


func _ready() -> void:
	can_sleep = false
	contact_monitor = true
	max_contacts_reported = 4
	linear_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
	angular_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
	angular_damp = 0.25
	var collider := CollisionShape3D.new()
	var shape := CylinderShape3D.new()
	shape.height = HALF_HEIGHT * 2.0
	shape.radius = 0.30
	collider.shape = shape
	add_child(collider)


func reset_body(origin: Vector3, velocity: Vector3, tilt: Vector2, fuel_fraction: float,
		payload_mass: float, engine_efficiency: float) -> void:
	fuel = fuel_fraction * FUEL_CAPACITY
	loaded_fuel = fuel
	payload = payload_mass
	efficiency = engine_efficiency
	var pose := Transform3D(Basis.from_euler(Vector3(tilt.x, 0, tilt.y)), origin)
	transform = pose
	linear_velocity = velocity
	angular_velocity = Vector3.ZERO
	# Synchronize the physics server before requesting the reset observation.
	var rid := get_rid()
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_TRANSFORM, pose)
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, velocity)
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, Vector3.ZERO)
	mass = DRY_MASS + payload + fuel


func thrust(action: Array, wind: Vector3, delta: float) -> void:
	var throttle := (clampf(float(action[0]), -1, 1) + 1.0) * 0.5
	var burnt := minf(fuel, BURN_RATE * throttle * delta)
	fuel -= burnt
	mass = DRY_MASS + payload + fuel
	var direction := Vector3(
		clampf(float(action[1]), -1, 1) * GIMBAL, 1.0,
		clampf(float(action[2]), -1, 1) * GIMBAL
	).normalized()
	apply_central_force(wind + global_basis * direction * (THRUST * efficiency * burnt / (BURN_RATE * delta)))
	# A separate inner-loop attitude stabilizer holds an upright body.
	var tilt_error := global_basis.y.cross(Vector3.UP)
	var stabilization := 2.8 * tilt_error - 1.4 * angular_velocity
	apply_torque(stabilization.limit_length(3.0))
