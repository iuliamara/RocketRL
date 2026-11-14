extends RefCounted

const PAD_HALF_WIDTH := 1.65
const WORLD_LIMIT := 14.0
const MAX_TICKS := 900
const GRAVITY := 3.2
const MIN_UP_DOT := 0.966


static func outcome(position: Vector3, up: Vector3, pre_impact_velocity: Vector3,
		omega: Vector3, touching: bool, ticks: int) -> String:
	if touching:
		var on_pad := absf(position.x) <= PAD_HALF_WIDTH and absf(position.z) <= PAD_HALF_WIDTH
		var horizontal_speed := Vector2(pre_impact_velocity.x, pre_impact_velocity.z).length()
		var stable := horizontal_speed <= 0.75 and absf(pre_impact_velocity.y) <= 1.15
		stable = stable and up.y >= MIN_UP_DOT and omega.length() <= 0.8
		if not stable:
			return "hard_contact"
		return "landing" if on_pad else "off_pad"
	if absf(position.x) > WORLD_LIMIT or absf(position.z) > WORLD_LIMIT or position.y > 22.0:
		return "out_of_bounds"
	return "time_limit" if ticks >= MAX_TICKS else ""


static func state_cost(position: Vector3, velocity: Vector3, up: Vector3) -> float:
	var horizontal_position := Vector2(position.x, position.z).length()
	var horizontal_speed := Vector2(velocity.x, velocity.z).length()
	var height := maxf(position.y - 0.8, 0)
	# Near the pad, a controlled descent should be slower than at altitude.
	var target_rate := -minf(2.0, 0.55 + 0.45 * height)
	return 1.8 * horizontal_position + 0.7 * horizontal_speed + 0.72 * height + 0.8 * absf(velocity.y - target_rate) + (1.0 - up.y)


static func terminal_reward(reason: String) -> float:
	match reason:
		"landing":
			return 24.0
		"off_pad":
			return -10.0
		"hard_contact", "out_of_bounds":
			return -16.0
		_:
			return 0.0
