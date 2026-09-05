
# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete and pifroggi on Discord

# Guided Color Fix model and architecture by Bendel https://huggingface.co/labx

import os
import sys
import vapoursynth as vs
from .utils import expression, tensorrt_inference, pad

core = vs.core


def _vsmlrt_inference(clip, ref, backend="ncnn", num_streams=1, gpu_id=0, engine_folder=None):
    # choose model and vsmlrt backend, then inference
    
    clip_w        = clip.width
    clip_h        = clip.height
    num_planes    = clip.format.num_planes
    precision     = clip.format.bits_per_sample
    force_rebuild = False
    opt_lvl       = 3
    opt_tile      = False
    func_name     = ".guided"
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    model_file    = "guidedcolorfix_ncnn.onnx" if backend in ["ncnn"] else f"guidedcolorfix_fp{precision}.onnx"  # ncnn backend needs fp32 onnx, will be converted to fp16 internally
    onnx_path     = os.path.join(current_dir, "models", model_file)
    engine_dir    = os.path.join(current_dir, "engines") if engine_folder is None else os.path.abspath(engine_folder)
    
    # ncnn model needs input to be mod4
    pad_w = (-clip_w) % 4
    pad_h = (-clip_h) % 4
    if backend in ["ncnn"] and (pad_w or pad_h):
        clip = pad(clip, pad_w, pad_h)
        ref  = pad(ref,  pad_w, pad_h)
    
    # inference
    input_clips = [clip, ref]
    if backend in ["tensorrt", "trt"]:
        out = tensorrt_inference(input_clips, onnx_path=onnx_path, engine_dir=engine_dir, clip_w=clip_w, clip_h=clip_h, num_planes=num_planes, precision=precision, opt_lvl=opt_lvl, opt_tile=opt_tile, num_streams=num_streams, gpu_id=gpu_id, force_rebuild=force_rebuild, func_name=func_name)
    elif backend in ["directml", "dml"]:
        out = core.ort.Model(input_clips, network_path=onnx_path, provider="DML", verbosity=1, device_id=gpu_id, num_streams=num_streams)  # verbosity=1 to silence constant fold warnings
    elif backend in ["ncnn"]:
        out = core.ncnn.Model(input_clips, network_path=onnx_path, fp16=clip.format.bits_per_sample == 16, output_format=1 if clip.format.bits_per_sample == 16 else 0, device_id=gpu_id, num_streams=num_streams)  # fp16=true/output_format=1 to allow fp16 input/output
    elif backend in ["cpu"]:
        out = core.ov.Model(input_clips, network_path=onnx_path, device="CPU")
    else:
        raise ValueError(f"vs_colorfix.guided: Backend must be CPU, NCNN, DirectML, or TensorRT.")

    # crop if needed and return
    if backend in ["ncnn"] and (pad_w or pad_h):
        return core.std.CropAbs(out, width=clip_w, height=clip_h)
    return out


def guided_color_fix(clip, ref, planes=None, backend="tensorrt", num_streams=1, gpu_id=0, engine_folder=None):
    """Fixes color shift based on a reference clip. This approach is guided by a trained AI model that can intelligently transfer colors while avoiding the typical
        bleed/bloom that can occur in the Average or Wavelet Color Fix, but is much slower.

    Args:
        clip: Clip where the color fix will be applied to. Must be in float format.
        ref: Reference clip where the colors are taken from. Should match the base clip somewhat. Compression, grain, or lower resolution are all okay.
        planes: Which planes to color fix. Any unmentioned planes will simply be copied. None means all planes will be color fixed.
        backend: The backend used to run the color fix.
            - `cpu` = CPU mode (very slow).
            - `ncnn` = GPU mode using NCNN. Works on almost any GPU, even Mac (fast).
            - `directml` = GPU mode using DirectML. Works on most GPUs, but Windows only (faster).
            - `tensorrt` = GPU mode using TensorRT. Requires an Nvidia RTX GPU (very fast).
        num_streams: Number of parallel GPU streams. Higher can be faster, but requires more VRAM. Does not affect the CPU backend.
        gpu_id: GPU index ID starting from 0 for the first compatible GPU. For example to switch between iGPU/dGPU. Does not affect the CPU backend.
        engine_folder: Optional path to the TensorRT engine storage location. By default engines are stored in `vs_colorfix/engines`. Only affects the TensorRT backend.
    """
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_colorfix.guided: Clip must be a vapoursynth clip.")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("vs_colorfix.guided: Ref must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_colorfix.guided: Clip must have constant format and dimensions.")
    if ref.format.id == vs.PresetVideoFormat.NONE or ref.width == 0 or ref.height == 0:
        raise TypeError("vs_colorfix.guided: Ref must have constant format and dimensions.")
    if clip.format.sample_type != vs.FLOAT:
        raise ValueError("vs_colorfix.guided: Input clips must be in float format.")
    if clip.format.id != ref.format.id:
        raise ValueError("vs_colorfix.guided: Clip and ref must have the same format.")
    if clip.num_frames != ref.num_frames:
        raise ValueError("vs_colorfix.guided: Clip and ref must have the same number of frames.")
    if not isinstance(gpu_id, int) or isinstance(gpu_id, bool):
        raise TypeError("vs_colorfix.guided: GPU ID must be an integer.")
    if gpu_id < 0:
        raise ValueError("vs_colorfix.guided: GPU ID can not be negative.")
    if not isinstance(num_streams, int) or isinstance(num_streams, bool):
        raise TypeError("vs_colorfix.guided: Number of parallel GPU streams (num_streams) must be an integer.")
    if num_streams < 1:
        raise ValueError("vs_colorfix.guided: Number of parallel GPU streams (num_streams) must be at least 1.")
    if not isinstance(backend, str):
        raise TypeError("vs_colorfix.guided: Backend must be a string.")
    
    orig_clip   = clip
    clip_format = clip.format
    num_planes  = clip_format.num_planes
    backend     = backend.lower()
    format_rgb  = vs.RGBS if backend == "cpu" else vs.RGBH
    
    if backend in ["directml", "dml"] and sys.platform != "win32":
        raise RuntimeError("vs_colorfix.guided: The DirectML backend is only available on Windows.")
    if backend in ["tensorrt", "trt"] and sys.platform not in ("win32", "linux"):
        raise RuntimeError("vs_colorfix.guided: The TensorRT backend is only available on Windows and Linux.")
    
    if planes is None:
        planes = list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if num_planes == 1:
        planes = [0]
    planes = set(planes)
    if not planes <= set(range(num_planes)):
        raise ValueError("vs_colorfix.guided: Invalid plane index specified.")
    
    # if rgb, do color fix directly
    if clip_format.color_family == vs.RGB:
        
        # resize ref if needed
        if ref.width != clip.width or ref.height != clip.height:
            ref = core.resize.Bilinear(ref, width=clip.width, height=clip.height)
        
        # clamp and convert to rgbs for cpu
        clip = expression(clip, expr="x 0 max 1 min", format=format_rgb if clip_format.id != format_rgb else None)
        ref  = expression(ref,  expr="x 0 max 1 min", format=format_rgb if clip_format.id != format_rgb else None)
        
        # inference
        fixed = _vsmlrt_inference(clip, ref, backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)
        
        # convert back, select planes, return
        if clip_format.id != format_rgb:
            fixed = core.resize.Point(fixed, format=clip_format)
        if len(planes) != num_planes:
            fixed = core.std.ShufflePlanes([fixed if i in planes else orig_clip for i in range(3)], [0, 1, 2], vs.RGB)
        return fixed
    
    # else get props for rgb roundtrip
    props_clip      = clip.get_frame(0).props
    props_ref       = ref.get_frame(0).props
    matrix          = props_clip.get("_Matrix", vs.MATRIX_BT709)
    matrix          = vs.MATRIX_BT709 if matrix == vs.MATRIX_UNSPECIFIED else matrix
    chroma_loc_clip = vs.ChromaLocation(props_clip.get("_ChromaLocation", vs.CHROMA_LEFT))
    chroma_loc_ref  = vs.ChromaLocation(props_ref.get("_ChromaLocation",  vs.CHROMA_LEFT))
    range_prop      = "_Range" if vs.__version__.release_major >= 74 else "_ColorRange"
    format_444      = clip_format.replace(subsampling_w=0, subsampling_h=0).id
    
    def convert_and_inference(clip, ref):
        # convert to rgb and resize ref
        clip = core.resize.Bilinear(clip, format=format_rgb)
        ref  = core.resize.Bilinear(ref,  format=format_rgb, width=clip.width, height=clip.height)
        
        # clamp
        clip = expression(clip, expr="x 0 max 1 min")
        ref  = expression(ref,  expr="x 0 max 1 min")
        
        # inference and convert back
        clip = _vsmlrt_inference(clip, ref, backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)
        return core.resize.Point(clip, format=format_444, matrix=matrix, range_s="full")

    def luma_to_chroma(clip, chroma_loc):
        # downscale luma to chroma resolution and shift to make 444
        y, u, v = core.std.SplitPlanes(clip)
        sx, sy = 1 << clip_format.subsampling_w, 1 << clip_format.subsampling_h
        x = -(sx - 1) / 2 if chroma_loc in (vs.CHROMA_LEFT, vs.CHROMA_TOP_LEFT, vs.CHROMA_BOTTOM_LEFT) else 0
        yoff = -(sy - 1) / 2 if chroma_loc in (vs.CHROMA_TOP_LEFT, vs.CHROMA_TOP) else (sy - 1) / 2 if chroma_loc in (vs.CHROMA_BOTTOM_LEFT, vs.CHROMA_BOTTOM) else 0
        y = core.resize.Bicubic(y, u.width, u.height, src_left=x, src_top=yoff, src_width=clip.width, src_height=clip.height)
        return core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)
    
    # set props for rgb roundtrip
    clip = core.std.SetFrameProps(clip, _ChromaLocation=chroma_loc_clip, _Matrix=matrix, **{range_prop: vs.RANGE_FULL})  # just assume full because model behaves the same and then copy original props on output
    ref  = core.std.SetFrameProps(ref,  _ChromaLocation=chroma_loc_ref,  _Matrix=matrix, **{range_prop: vs.RANGE_FULL})
    
    # if gray, convert to rgb and color fix
    if clip_format.color_family == vs.GRAY:
        fixed = convert_and_inference(clip, ref)
        return core.std.CopyFrameProps(fixed, orig_clip)
    
    # if yuv444, color fix, select planes, return
    if not (clip_format.subsampling_w or clip_format.subsampling_h):
        fixed = convert_and_inference(clip, ref)
        if len(planes) != num_planes:
            fixed = core.std.ShufflePlanes([fixed if i in planes else orig_clip for i in range(3)], [0, 1, 2], vs.YUV, prop_src=orig_clip)
        else:
            fixed = core.std.CopyFrameProps(fixed, orig_clip)
        return fixed

    # if yuv subsampled, fix separately, select planes and return
    fixed    = convert_and_inference(clip, ref) if 0 in planes else None
    fixed_uv = convert_and_inference(luma_to_chroma(clip, chroma_loc_clip), luma_to_chroma(ref, chroma_loc_ref)) if planes & {1, 2} else None
    return core.std.ShufflePlanes([fixed if 0 in planes else orig_clip, fixed_uv if 1 in planes else orig_clip, fixed_uv if 2 in planes else orig_clip], [0, 1, 2], vs.YUV, prop_src=orig_clip)
