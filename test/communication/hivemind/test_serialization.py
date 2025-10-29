from genrl.communication.communication import Payload
from genrl.state.game_tree import WorldState
import msgpack
import pytest
from genrl.serialization.msgpack_utils import _msgpack_encode, _msgpack_decode

def test_payload_serialization():
    """Test that Payload and WorldState objects can be serialized and deserialized correctly."""
    # Create test data
    world_state = WorldState(
        environment_states={"question": "What is 2+2?", "answer": "4"},
        opponent_states=None,
        personal_states={"iteration": 1, "score": 0.95}
    )
    
    payload = Payload(
        world_state=world_state,
        actions=["action1", "action2"],
        metadata={"timestamp": 12345, "agent_id": "agent_001"}
    )
    
    # Serialize using msgpack with custom encoder
    packer = msgpack.Packer(use_bin_type=True, default=_msgpack_encode)
    serialized_bytes = packer.pack(payload)
    
    # Verify we got bytes
    assert isinstance(serialized_bytes, bytes)
    assert len(serialized_bytes) > 0
    
    # Deserialize using msgpack with custom decoder
    unpacker = msgpack.Unpacker(raw=False, object_hook=_msgpack_decode, max_buffer_size=100*1024*1024)
    unpacker.feed(serialized_bytes)
    deserialized_payload = next(unpacker)
    
    # Verify the deserialized object is a Payload
    assert isinstance(deserialized_payload, Payload)
    
    # Verify world_state was reconstructed correctly
    assert isinstance(deserialized_payload.world_state, WorldState)
    assert deserialized_payload.world_state.environment_states == {"question": "What is 2+2?", "answer": "4"}
    assert deserialized_payload.world_state.opponent_states is None
    assert deserialized_payload.world_state.personal_states == {"iteration": 1, "score": 0.95}
    
    # Verify actions were reconstructed correctly
    assert deserialized_payload.actions == ["action1", "action2"]
    
    # Verify metadata was reconstructed correctly
    assert deserialized_payload.metadata == {"timestamp": 12345, "agent_id": "agent_001"}


def test_encode_arbitrary_object_raises_error():
    """Test that encoding an arbitrary object raises TypeError."""
    class ArbitraryObject:
        def __init__(self):
            self.data = "test"
    
    arbitrary_obj = ArbitraryObject()
    packer = msgpack.Packer(use_bin_type=True, default=_msgpack_encode)
    
    # Should raise TypeError when trying to serialize an arbitrary object
    with pytest.raises(TypeError, match="Object of type ArbitraryObject is not msgpack serializable"):
        packer.pack(arbitrary_obj)


def test_decode_unknown_type_raises_error():
    """Test that decoding an unknown __type__ raises TypeError."""
    # Create a dict with an unknown __type__
    unknown_type_dict = {
        "__type__": "UnknownType",
        "some_field": "some_value"
    }
    
    # Should raise TypeError when encountering unknown __type__
    with pytest.raises(TypeError, match="Unknown __type__ 'UnknownType' in deserialization"):
        _msgpack_decode(unknown_type_dict)


def test_decode_invalid_payload_raises_error():
    """Test that decoding invalid Payload data raises ValueError."""
    # Create a dict with unexpected keyword arguments that will cause Payload instantiation to fail
    # Dataclasses raise TypeError when given unexpected keyword arguments
    invalid_payload_dict = {
        "__type__": "Payload",
        "world_state": None,
        "actions": None,
        "metadata": None,
        "unexpected_field": "this will cause failure",  # Extra field not in dataclass
    }
    
    # This should raise ValueError when Payload instantiation fails
    with pytest.raises(ValueError, match="Failed to instantiate Payload"):
        _msgpack_decode(invalid_payload_dict)


def test_decode_invalid_worldstate_raises_error():
    """Test that decoding invalid WorldState data raises ValueError."""
    # Create a dict with unexpected keyword arguments that will cause WorldState instantiation to fail
    # Dataclasses raise TypeError when given unexpected keyword arguments
    invalid_worldstate_dict = {
        "__type__": "WorldState",
        "environment_states": None,
        "opponent_states": None,
        "personal_states": None,
        "invalid_extra_field": "this will cause failure",  # Extra field not in dataclass
    }
    
    # This should raise ValueError when WorldState instantiation fails
    with pytest.raises(ValueError, match="Failed to instantiate WorldState"):
        _msgpack_decode(invalid_worldstate_dict)

