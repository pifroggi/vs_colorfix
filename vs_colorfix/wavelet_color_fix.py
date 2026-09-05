
# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete and pifroggi on Discord

# Wavelet Color Fix idea from sd-webui-stablesr https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py

import os
import sys
import warnings
import vapoursynth as vs
from .utils import expression, compensate_wavelets, inference_groups, tensorrt_inference

core = vs.core


def _vsmlrt_inference(clips, refs, wavelets, backend="ncnn", num_streams=1, gpu_id=0, engine_folder=None):
    # choose model and vsmlrt backend, then inference
    
    base_clip     = clips[0]
    clip_w        = base_clip.width
    clip_h        = base_clip.height
    num_planes    = base_clip.format.num_planes
    precision     = 32 if backend in ["ncnn"] else base_clip.format.bits_per_sample  # ncnn backend needs fp32 onnx, will be converted to fp16 internally
    force_rebuild = False
    opt_lvl       = 5
    opt_tile      = True
    func_name     = ".wavelet"
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    model_file    = f"waveletcolorfix_w{wavelets}_c{num_planes}_fp{precision}.onnx"
    onnx_path     = os.path.join(current_dir, "models", model_file)
    engine_dir    = os.path.join(current_dir, "engines") if engine_folder is None else os.path.abspath(engine_folder)

    # interleave all input clips
    input_clips = [core.std.Interleave(clips), core.std.Interleave(refs)] if len(clips) > 1 else [clips[0], refs[0]]
    
    # inference
    if backend in ["tensorrt", "trt"]:
        out = tensorrt_inference(input_clips, onnx_path=onnx_path, engine_dir=engine_dir, clip_w=clip_w, clip_h=clip_h, num_planes=num_planes, precision=precision, opt_lvl=opt_lvl, opt_tile=opt_tile, num_streams=num_streams, gpu_id=gpu_id, force_rebuild=force_rebuild, func_name=func_name)
    elif backend in ["directml", "dml"]:
        out = core.ort.Model(input_clips, network_path=onnx_path, provider="DML", device_id=gpu_id, num_streams=num_streams)
    elif backend in ["ncnn"]:
        out = core.ncnn.Model(input_clips, network_path=onnx_path, fp16=base_clip.format.bits_per_sample == 16, output_format=1 if base_clip.format.bits_per_sample == 16 else 0, device_id=gpu_id, num_streams=num_streams)  # fp16=true/output_format=1 to allow fp16 input/output
    else:
        raise ValueError("vs_colorfix.wavelet: Backend must be CPU, NCNN, DirectML, or TensorRT.")
    
    # vsmlrt outputs yuv as rgb, recombine as yuv if input was yuv
    if base_clip.format.color_family == vs.YUV and out.format.color_family != vs.YUV:
        out = core.std.ShufflePlanes(out, planes=[0, 1, 2], colorfamily=base_clip.format.color_family, prop_src=base_clip)
    
    # separate clips and return
    return [core.std.SelectEvery(out, cycle=len(clips), offsets=i) for i in range(len(clips))] if len(clips) > 1 else [out]


def _wavelet_color_fix_vsmlrt(clip, ref, wavelets, planes, backend="ncnn", num_streams=1, gpu_id=0, engine_folder=None):
    # gpu backends using vsmlrt
    
    if clip.format.sample_type != vs.FLOAT:
        raise ValueError("vs_colorfix.wavelet: Input clips must be in float format when using a GPU backend. Use 16-bit float for best performance, if supported by your GPU. Most modern GPUs do.")
    
    clip_format    = clip.format
    num_planes     = clip.format.num_planes
    plane_wavelets = compensate_wavelets(clip_format, wavelets)
    
    # if all planes can use the same model and size, inference as one
    if planes == set(range(num_planes)) and clip_format.subsampling_w == 0 and clip_format.subsampling_h == 0 and len(set(plane_wavelets)) == 1:
        return _vsmlrt_inference([clip], [ref], wavelets=plane_wavelets[0], backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)[0]
    
    # else inference selected planes separately, interleaved when possible
    clips = list(core.std.SplitPlanes(clip))
    refs  = list(core.std.SplitPlanes(ref))
    
    for effective_wavelets, selected_planes in inference_groups(clips, planes, plane_wavelets):
        selected_out = _vsmlrt_inference([clips[p] for p in selected_planes], [refs[p] for p in selected_planes], wavelets=effective_wavelets, backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)
        for p, processed in zip(selected_planes, selected_out):
            clips[p] = processed
    
    return core.std.ShufflePlanes(clips, [0] * num_planes, clip_format.color_family)


def _wavelet_color_fix_atwt(clip, ref, wavelets, planes):
    # cpu backend using the vapoursynth-atwt plugin

    clip_format    = clip.format
    num_planes     = clip.format.num_planes
    plane_wavelets = compensate_wavelets(clip_format, wavelets)
    
    def _decompose(base, wavelet_count):
        details = []
        for radius in range(1, wavelet_count + 1):
            detail = core.atwt.ExtractFrequency(base, radius=radius)
            base = core.std.MakeDiff(base, detail)
            details.append(detail)
        return details + [base]
    
    def _recombine(layers):
        out = layers[-1]
        for detail in reversed(layers[:-1]):
            out = core.atwt.ReplaceFrequency(base=out, detail=detail)
        return out
    
    def _fix_clip(clip, ref, wavelet_count):
        c_layers = _decompose(clip, wavelet_count)
        r_layers = _decompose(ref, wavelet_count)
        return _recombine(c_layers[:-1] + [r_layers[-1]])
    
    # process the whole clip at once if possible
    if planes == set(range(num_planes)) and clip_format.subsampling_w == 0 and clip_format.subsampling_h == 0 and len(set(plane_wavelets)) == 1:
        return _fix_clip(clip, ref, plane_wavelets[0])
    
    # else each plane separately
    clips = list(core.std.SplitPlanes(clip))
    refs  = list(core.std.SplitPlanes(ref))
    
    for p in sorted(planes):
        clips[p] = _fix_clip(clips[p], refs[p], plane_wavelets[p])
    
    return core.std.ShufflePlanes(clips, [0] * num_planes, clip_format.color_family)


def wavelet_color_fix(clip, ref, wavelets=4, planes=None, backend="ncnn", num_streams=2, gpu_id=0, engine_folder=None):
    """Fixes color shift based on a reference clip. Works similarly to `average()`, but more accurate when color differences are large, at the cost of more computation.

    Args:
        clip: Clip where the color fix will be applied to. Any format on CPU, must be float format on GPU.
        ref: Reference clip where the colors are taken from. Should match the base clip somewhat. Compression, grain, or lower resolution are all okay.
        wavelets: Number of wavelets, around 4 seems to work best in most cases. Higher means a more global color match and wider bloom/bleed. Lower means a more 
            local color match and smaller bloom/bleed. Lower is also faster. Too low and the reference clip will become visible. Test values 3 and 8 and this will become more clear.
        planes: Which planes to color fix. Any unmentioned planes will simply be copied. None means all planes will be color fixed.
        backend: The backend used to run the color fix.
            - `cpu` = CPU mode (slow).
            - `ncnn` = GPU mode using NCNN. Works on almost any GPU, even Mac (fast).
            - `directml` = GPU mode using DirectML. Works on most GPUs, but Windows only (fast).
            - `tensorrt` = GPU mode using TensorRT. Requires an Nvidia RTX GPU (very fast).
        num_streams: Number of parallel GPU streams. Higher can be faster, but requires more VRAM. Does not affect the CPU backend.
        gpu_id: GPU index ID starting from 0 for the first compatible GPU. For example to switch between iGPU/dGPU. Does not affect the CPU backend.
        engine_folder: Optional path to the TensorRT engine storage location. By default engines are stored in `vs_colorfix/engines`. Only affects the TensorRT backend.
    """
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_colorfix.wavelet: Clip must be a vapoursynth clip.")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("vs_colorfix.wavelet: Ref must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_colorfix.wavelet: Clip must have constant format and dimensions.")
    if ref.format.id == vs.PresetVideoFormat.NONE or ref.width == 0 or ref.height == 0:
        raise TypeError("vs_colorfix.wavelet: Ref must have constant format and dimensions.")
    if clip.format.id != ref.format.id:
        raise ValueError("vs_colorfix.wavelet: Clip and ref must have the same format.")
    if clip.num_frames != ref.num_frames:
        raise ValueError("vs_colorfix.wavelet: Clip and ref must have the same number of frames.")
    if not isinstance(wavelets, int) or isinstance(wavelets, bool):
        raise TypeError("vs_colorfix.wavelet: Number of wavelets must be an integer.")
    if not 1 <= wavelets <= 10:
        raise ValueError("vs_colorfix.wavelet: Number of wavelets must be in the range 1-10.")
    if not isinstance(gpu_id, int) or isinstance(gpu_id, bool):
        raise TypeError("vs_colorfix.wavelet: GPU ID must be an integer.")
    if gpu_id < 0:
        raise ValueError("vs_colorfix.wavelet: GPU ID can not be negative.")
    if not isinstance(num_streams, int) or isinstance(num_streams, bool):
        raise TypeError("vs_colorfix.wavelet: Number of parallel GPU streams (num_streams) must be an integer.")
    if num_streams < 1:
        raise ValueError("vs_colorfix.wavelet: Number of parallel GPU streams (num_streams) must be at least 1.")
    if not isinstance(backend, str):
        raise TypeError("vs_colorfix.wavelet: Backend must be a string.")
    if clip.format.bits_per_sample <= 8 or ref.format.bits_per_sample <= 8:
        warnings.warn("vs_colorfix.wavelet: Input clips have a low bit depth, which will cause banding. 16-bit input is recommended.", UserWarning, stacklevel=2)
    
    clip_format = clip.format
    num_planes  = clip.format.num_planes
    backend     = backend.lower()
    format_proc = clip_format.replace(bits_per_sample=32 if backend == "cpu" else 16) if clip_format.sample_type == vs.FLOAT else clip_format  # use float32 or keep int for atwt cpu backend, float16 for gpu backends
    req_convert = clip_format.id != format_proc.id
    
    if backend in ["directml", "dml"] and sys.platform != "win32":
        raise RuntimeError("vs_colorfix.wavelet: The DirectML backend is only available on Windows.")
    if backend in ["tensorrt", "trt"] and sys.platform not in ("win32", "linux"):
        raise RuntimeError("vs_colorfix.wavelet: The TensorRT backend is only available on Windows and Linux.")
    if backend in ["cpu"] and not hasattr(core, "atwt"):
        raise RuntimeError("vs_colorfix.wavelet: Please install the plugin 'Vapoursynth-ATWT' to use the CPU backend.")
    
    if planes is None:
        planes = list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if num_planes == 1:
        planes = [0]
    planes = set(planes)
    if not planes <= set(range(num_planes)):
        raise ValueError("vs_colorfix.wavelet: Invalid plane index specified.")
    
    # resize ref if needed
    if ref.width != clip.width or ref.height != clip.height:
        ref = core.resize.Bilinear(ref, width=clip.width, height=clip.height)
    
    # clamp, shift uv, convert precision if needed
    shift_uv   = False
    copy_plane = "x" if req_convert else ""  # empty expressions can not copy planes when changing formats
    if clip_format.sample_type == vs.FLOAT:
        clamp_expr = "x 0 max 1 min"
        if clip_format.color_family == vs.YUV:
            clamp_uv_expr = "x -0.5 max 0.5 min"
            if backend != "cpu":
                shift_uv = any(p in planes for p in (1, 2))
                clamp_uv_expr += " 0.5 +"  # shift uv to be within 0-1 for model input
            expr = [clamp_expr if 0 in planes else copy_plane, clamp_uv_expr if 1 in planes else copy_plane, clamp_uv_expr if 2 in planes else copy_plane]
        else:
            expr = [clamp_expr if p in planes else copy_plane for p in range(num_planes)]
        
        clip = expression(clip, expr=expr, format=format_proc if req_convert else None)
        ref  = expression(ref,  expr=expr, format=format_proc if req_convert else None)
    
    # color fix
    if backend == "cpu":
        clip = _wavelet_color_fix_atwt(clip, ref, wavelets=wavelets, planes=planes)
    else:
        clip = _wavelet_color_fix_vsmlrt(clip, ref, wavelets=wavelets, planes=planes, backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)
    
    # undo uv shift and conversion
    if shift_uv:
        clip = expression(clip, expr=[copy_plane, "x 0.5 -" if 1 in planes else copy_plane, "x 0.5 -" if 2 in planes else copy_plane], format=clip_format if req_convert else None)
    elif req_convert:
        clip = core.resize.Point(clip, format=clip_format)
    return clip
