"""Measured ZIP packets: features plus explicitly transmitted metadata/thumbnail."""
import io
import json
import zipfile
import numpy as np
import torch
import torch.nn.functional as F

def pack_features(features, bits):
    values = np.asarray(features,dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite input features")
    if bits == 16:
        half = values.astype("<f2")
        if not np.isfinite(half).all():
            raise ValueError("Features exceed FP16 storage range")
        return half.tobytes(), b"", {"encoding":"fp16"}
    if bits not in (4,8):
        raise ValueError("Only 4,8,16 bits supported")
    limit = 127 if bits == 8 else 7
    scale = np.maximum(np.max(np.abs(values),axis=0),1e-8) / limit
    # Scales are actual FP32 transmitted values, not estimated overhead.
    scale = scale.astype("<f4")
    q = np.clip(np.rint(values/scale),-limit,limit).astype(np.int8)
    if bits == 8:
        payload = q.tobytes()
    else:
        unsigned = (q.reshape(-1).astype(np.int16)+7).astype(np.uint8)
        if unsigned.size % 2:
            unsigned = np.pad(unsigned,(0,1),constant_values=7)
        payload = (unsigned[::2] | (unsigned[1::2] << 4)).tobytes()
    return payload, scale.tobytes(), {"encoding":"symmetric_int"+str(bits),"scale_dtype":"float32"}

def unpack_features(payload, scales, shape, encoding):
    if encoding == "fp16":
        return np.frombuffer(payload,dtype="<f2").astype(np.float32).reshape(shape)
    scale=np.frombuffer(scales,dtype="<f4")
    if encoding == "symmetric_int8":
        q=np.frombuffer(payload,dtype=np.int8).reshape(shape)
    elif encoding == "symmetric_int4":
        packed=np.frombuffer(payload,dtype=np.uint8)
        unsigned=np.empty(packed.size*2,dtype=np.uint8)
        unsigned[::2],unsigned[1::2]=packed & 15,packed >> 4
        q=(unsigned[:int(np.prod(shape))].astype(np.int16)-7).reshape(shape)
    else:
        raise ValueError("Unknown feature encoding")
    return q.astype(np.float32)*scale

def pool_tokens(features, original_grid, pool_side):
    tensor=torch.from_numpy(features).reshape(*original_grid,-1).permute(2,0,1)[None]
    pooled=F.adaptive_avg_pool2d(tensor.float(),(pool_side,pool_side))
    return pooled[0].permute(1,2,0).reshape(pool_side*pool_side,-1).numpy()

def encode_packet(features, original_grid, bits, pool_side, sigma, seed, model_revision, thumbnail=b""):
    pooled=features if pool_side == original_grid[0] else pool_tokens(features,original_grid,pool_side)
    payload,scales,details=pack_features(pooled,bits)
    metadata={"schema":"iris-final-head-pilot-v1","checkpoint_revision":model_revision,
              "layer":24,"sigma":sigma,"seed":seed,"noise_rng":"torch-cpu-2.10.0-normal",
              "image_size":[original_grid[0]*16,original_grid[1]*16],
              "original_grid":list(original_grid),"feature_grid":[pool_side,pool_side],
              "shape":list(pooled.shape),"feature_interpolation":"bilinear_align_corners_false",
              "thumbnail":bool(thumbnail),**details}
    pieces={"metadata.json":json.dumps(metadata,separators=(",",":"),sort_keys=True).encode(),
            "features.bin":payload,"scales.bin":scales}
    if thumbnail:
        pieces["thumbnail.jpg"]=thumbnail
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in pieces.items():
            info=zipfile.ZipInfo(name,date_time=(2026,10,13,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data,compresslevel=6)
    packet=stream.getvalue()
    return packet, {"total_bytes":len(packet),"uncompressed_component_bytes":{k:len(v) for k,v in pieces.items()},
                    "metadata":metadata}

def decode_packet(packet):
    with zipfile.ZipFile(io.BytesIO(packet)) as archive:
        metadata=json.loads(archive.read("metadata.json"))
        if metadata["schema"] != "iris-final-head-pilot-v1" or metadata["layer"] != 24:
            raise ValueError("Unsupported packet")
        features=unpack_features(archive.read("features.bin"),archive.read("scales.bin"),
                                 metadata["shape"],metadata["encoding"])
        tensor=torch.from_numpy(features).reshape(*metadata["feature_grid"],-1).permute(2,0,1)[None]
        tensor=F.interpolate(tensor,size=metadata["original_grid"],mode="bilinear",align_corners=False)
        features=tensor.permute(0,2,3,1).reshape(1,-1,metadata["shape"][-1])
        thumbnail=archive.read("thumbnail.jpg") if metadata["thumbnail"] else None
    return metadata,features,thumbnail
