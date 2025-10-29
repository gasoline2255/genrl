from genrl.communication.communication import Payload
from genrl.state.game_tree import WorldState

def _msgpack_encode(obj):
    """Encoder for custom genrl dataclasses."""
    if isinstance(obj, Payload):
        return {
            "__type__": "Payload",
            "world_state": obj.world_state,
            "actions": obj.actions,
            "metadata": obj.metadata,
        }
    if isinstance(obj, WorldState):
        return {
            "__type__": "WorldState",
            "environment_states": obj.environment_states,
            "opponent_states": obj.opponent_states,
            "personal_states": obj.personal_states,
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not msgpack serializable")


def _msgpack_decode(obj):
    """Decoder for custom genrl dataclasses."""
    if isinstance(obj, dict) and "__type__" in obj:
        type_name = obj["__type__"]

        constructor_args = {k: v for k, v in obj.items() if k != "__type__"}
        
        if type_name == "Payload":
            try:
                return Payload(**constructor_args)
            except Exception as e:
                raise ValueError(f"Failed to instantiate Payload: {e}")
        elif type_name == "WorldState":
            try:
                return WorldState(**constructor_args)
            except Exception as e:
                raise ValueError(f"Failed to instantiate WorldState: {e}")
        else:
            raise TypeError(f"Unknown __type__ '{type_name}' in deserialization. Only 'Payload' and 'WorldState' are supported.")

    return obj