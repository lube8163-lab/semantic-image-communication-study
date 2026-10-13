import numpy as np
import pytest
from feature_packet import pack_features,unpack_features,encode_packet,decode_packet

@pytest.mark.parametrize("bits",[4,8,16])
def test_quantization_real_error_bound_and_odd_size(bits):
    rng=np.random.default_rng(81)
    x=rng.normal(size=(3,5)).astype(np.float32)
    payload,scales,details=pack_features(x,bits)
    y=unpack_features(payload,scales,x.shape,details["encoding"])
    assert y.shape==x.shape
    assert np.isfinite(y).all()
    if bits in (4,8):
        limit=7 if bits==4 else 127
        bound=np.max(np.abs(x),axis=0)/limit/2
        assert np.all(np.abs(x-y)<=bound+1e-6)
    else:
        assert np.max(np.abs(x-y))<0.002

def test_actual_packet_bytes_and_no_hidden_original():
    x=np.arange(16*8,dtype=np.float32).reshape(16,8)/100
    packet,account=encode_packet(x,(4,4),4,2,0.1,7,"fixedrevision",b"thumbnail-bytes")
    metadata,y,thumb=decode_packet(packet)
    assert account["total_bytes"]==len(packet)
    assert thumb==b"thumbnail-bytes"
    assert y.shape==(1,16,8)
    assert "image_id" not in metadata
    assert "input_sha256" not in metadata
    assert metadata["seed"]==7

def test_nonfinite_is_not_silently_serialized():
    with pytest.raises(ValueError,match="Non-finite"):
        pack_features(np.array([[np.nan]],dtype=np.float32),8)
