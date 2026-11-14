extends RefCounted

const LIMIT := 65536
var stream := StreamPeerTCP.new()


func connect_local(port: int) -> bool:
	if stream.connect_to_host("127.0.0.1", port) != OK:
		return false
	var deadline := Time.get_ticks_msec() + 5000
	while Time.get_ticks_msec() < deadline:
		stream.poll()
		if stream.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			stream.set_no_delay(true)
			return true
		if stream.get_status() == StreamPeerTCP.STATUS_ERROR:
			return false
		OS.delay_msec(5)
	return false


func receive() -> Dictionary:
	stream.poll()
	if stream.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return {}
	var header := stream.get_data(4)
	if header[0] != OK:
		return {}
	var size: int = header[1].decode_u32(0)
	if size <= 0 or size > LIMIT:
		return {}
	var payload := stream.get_data(size)
	if payload[0] != OK:
		return {}
	var message: Variant = JSON.parse_string(payload[1].get_string_from_utf8())
	return message if message is Dictionary else {}


func send(message: Dictionary) -> bool:
	var payload := JSON.stringify(message).to_utf8_buffer()
	var header := PackedByteArray([0, 0, 0, 0])
	header.encode_u32(0, payload.size())
	return stream.put_data(header + payload) == OK
