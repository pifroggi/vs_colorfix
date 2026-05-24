
# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete and pifroggi on Discord

# Wavelet Color Fix idea from sd-webui-stablesr https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py

import os
import shutil
import logging
import warnings
import subprocess
import vapoursynth as vs
from pathlib import Path
from .average_color_fix import _expression

core = vs.core


def _plane_wavelets(clip_format, wavelets):
    # compensate wavelet count for subsampled planes
    plane_wavelets = [wavelets] * clip_format.num_planes
    
    if clip_format.color_family == vs.YUV and clip_format.num_planes > 1:
        subsampling_shift = max(clip_format.subsampling_w, clip_format.subsampling_h)
        if subsampling_shift > 0:
            chroma_wavelets = max(1, wavelets - subsampling_shift)
            for p in range(1, clip_format.num_planes):
                plane_wavelets[p] = chroma_wavelets
    
    return plane_wavelets


def _inference_groups(clips, planes, plane_wavelets):
    # group planes by wavelet count and size so interleaved inference only combines compatible planes
    groups = {}
    for p in sorted(planes):
        plane_clip = clips[p]
        key = (plane_wavelets[p], plane_clip.width, plane_clip.height)
        if key not in groups:
            groups[key] = []
        groups[key].append(p)
    
    return [(key[0], grouped_planes) for key, grouped_planes in groups.items()]


def _get_trtexec():
    # first search for tensorrt plugins, then check for trtexec
    exe_name = "trtexec.exe" if os.name == "nt" else "trtexec"
    plugins_path = None
    
    try:
        info = core.trt.Version()
    except Exception as e:
        raise RuntimeError("vs_colorfix.wavelet: Please install a version of vs-mlrt with TensorRT support or choose a different backend.") from e
    
    path = info.get("path")
    
    # get plugin path
    if isinstance(path, bytes):
        path = path.decode(errors="ignore")
    if path:
        plugins_path = os.path.dirname(path)
    
    # try finding vsmlrt trtexec first, then check for system trtexec
    if plugins_path is not None:
        local_trtexec = Path(plugins_path) / "vsmlrt-cuda" / exe_name
        if local_trtexec.is_file() and os.access(str(local_trtexec), os.X_OK):
            return local_trtexec
    
    system_trtexec = shutil.which("trtexec")
    if system_trtexec is not None:
        return Path(system_trtexec)
    
    raise FileNotFoundError("vs_colorfix.wavelet: trtexec not found. Please install a version of vs-mlrt with TensorRT support or choose a different backend. Make sure to follow the installation instructions.")


def _get_engine(onnx_path, engine_dir, engine_w, engine_h, num_planes, precision, gpu_id=0, force_rebuild=False) -> str:
    # build or get path to tensorrt engine
    os.makedirs(engine_dir, exist_ok=True)  # create engine folder if needed
    model_name   = Path(onnx_path).stem
    engine_name  = f"{model_name}_h{engine_h}_w{engine_w}_gpu{gpu_id}.engine"
    engine_path  = os.path.join(engine_dir, engine_name)
    trtexec_path = _get_trtexec()
    
    # if engine file exist, return it
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path
    
    # else build new engine
    logging.warning("vs_colorfix.wavelet: Building new TensorRT engine for width=%d, height=%d and precision=fp%d. This may take a few minutes.", engine_w, engine_h, precision)
    opt_shapes = f"input:1x{num_planes*2}x{engine_h}x{engine_w}"
    cmd = [
        str(trtexec_path),
        "--stronglyTyped",
        f"--inputIOFormats=fp{precision}:chw",
        f"--outputIOFormats=fp{precision}:chw",
        "--skipInference",
        "--memPoolSize=workspace:4096",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--optShapes={opt_shapes}",
        f"--builderOptimizationLevel=5",
        f"--tilingOptimizationLevel=3",
        f"--device={gpu_id}",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = (
            "vs_colorfix.wavelet: trtexec failed while building the TensorRT engine.\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Return code: {e.returncode}\n"
        )
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e
    
    logging.warning("vs_colorfix.wavelet: Engine building complete.")
    return engine_path


def _tensorrt_inference(input_clips, onnx_path, engine_dir, clip_w, clip_h, num_planes, precision, num_streams=1, gpu_id=0, force_rebuild=False):
    engine_path = _get_engine(onnx_path=onnx_path, engine_dir=engine_dir, engine_w=clip_w, engine_h=clip_h, num_planes=num_planes, precision=precision, gpu_id=gpu_id, force_rebuild=force_rebuild)
    model_args  = dict(engine_path=engine_path, num_streams=num_streams, device_id=gpu_id)
    
    # try inference, rebuild engine if it fails
    try:
        out = core.trt.Model(input_clips, **model_args)
    except vs.Error as e:
        err_msg = str(e).lower()
        serialization_keywords = ("serialize", "serialization", "deserialize", "deserialization")
        if any(k in err_msg for k in serialization_keywords) and not force_rebuild:
            logging.warning("vs_colorfix.wavelet: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
            model_args["engine_path"] = _get_engine(onnx_path=onnx_path, engine_dir=engine_dir, engine_w=clip_w, engine_h=clip_h, num_planes=num_planes, precision=precision, gpu_id=gpu_id, force_rebuild=True)
            out = core.trt.Model(input_clips, **model_args)
        else:
            raise
    return out


def _vsmlrt_inference(clips, refs, wavelets, backend="ncnn", num_streams=1, gpu_id=0, engine_folder=None):
    # choose model and vsmlrt backend, then inference
    
    base_clip     = clips[0]
    clip_w        = base_clip.width
    clip_h        = base_clip.height
    num_planes    = base_clip.format.num_planes
    precision     = 32 if backend in ["ncnn"] else base_clip.format.bits_per_sample  # ncnn backend needs fp32 onnx, will be converted to fp16 internally
    force_rebuild = False
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    model_file    = f"waveletcolorfix_w{wavelets}_c{num_planes}_fp{precision}.onnx"
    onnx_path     = os.path.join(current_dir, "models", model_file)
    engine_dir    = os.path.join(current_dir, "engines") if engine_folder is None else os.path.abspath(engine_folder)

    # interleave all input clips
    input_clips = [core.std.Interleave(clips), core.std.Interleave(refs)] if len(clips) > 1 else [clips[0], refs[0]]
    
    # inference
    if backend in ["tensorrt", "trt"]:
        out = _tensorrt_inference(input_clips, onnx_path=onnx_path, engine_dir=engine_dir, clip_w=clip_w, clip_h=clip_h, num_planes=num_planes, precision=precision, num_streams=num_streams, gpu_id=gpu_id, force_rebuild=force_rebuild)
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
    
    if not isinstance(num_streams, int) or isinstance(num_streams, bool):
        raise TypeError("vs_colorfix.wavelet: Number of parallel GPU streams (num_streams) must be an integer.")
    if num_streams < 1:
        raise ValueError("vs_colorfix.wavelet: Number of parallel GPU streams (num_streams) must be at least 1.")
    if clip.format.sample_type != vs.FLOAT:
        raise ValueError("vs_colorfix.wavelet: Input clips must be in float format when using a GPU backend. Use 16-bit float for best performance, if supported by your GPU. Most modern GPUs do.")
    
    clip_format    = clip.format
    num_planes     = clip.format.num_planes
    plane_wavelets = _plane_wavelets(clip_format, wavelets)
    
    # if all planes can use the same model and size, inference as one
    if planes == set(range(num_planes)) and clip_format.subsampling_w == 0 and clip_format.subsampling_h == 0 and len(set(plane_wavelets)) == 1:
        return _vsmlrt_inference([clip], [ref], wavelets=plane_wavelets[0], backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)[0]
    
    # else inference selected planes separately, interleaved when possible
    clips = list(core.std.SplitPlanes(clip))
    refs  = list(core.std.SplitPlanes(ref))
    
    for effective_wavelets, selected_planes in _inference_groups(clips, planes, plane_wavelets):
        selected_out = _vsmlrt_inference([clips[p] for p in selected_planes], [refs[p] for p in selected_planes], wavelets=effective_wavelets, backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)
        for p, processed in zip(selected_planes, selected_out):
            clips[p] = processed
    
    return core.std.ShufflePlanes(clips, [0] * num_planes, clip_format.color_family)


def _wavelet_color_fix_atwt(clip, ref, wavelets, planes):
    # cpu backend using the vapoursynth-atwt plugin
    
    if clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample == 16:
        raise ValueError("vs_colorfix.wavelet: The CPU backend does not support 16-bit float formats. Consider using a GPU backend, or change formats.")
    
    clip_format    = clip.format
    num_planes     = clip.format.num_planes
    plane_wavelets = _plane_wavelets(clip_format, wavelets)
    
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
    """Fixes color shift based on a reference clip. Works similarly to `average()`, but more accurate with large color differences at the cost of more computation. Both clips must have close to the same content.

    Args:
        clip: Clip where the color fix will be applied to.
        ref: Reference clip where the colors are taken from.
        wavelets: Number of wavelets, 3-5 seems to work best in most cases. Higher means a more global color match and wider bloom/bleed. Lower means a more 
            local color match and smaller bloom/bleed. Lower is also faster. Too low and the reference clip will become visible. Test values 3 and 8 and this will become more clear.
        planes: Which planes to color fix. Any unmentioned planes will simply be copied. None means all planes will be color fixed.
        backend: The backend used to run the color fix. **16-bit float input is always much faster on GPU, but not supported by older GPUs.**
            - `cpu` = CPU mode using the Vapoursynth-ATWT plugin (slowest).
            - `ncnn` = GPU mode using vs-mlrt with NCNN support. Works on almost any GPU, even MAC (fast).
            - `directml` = GPU mode using vs-mlrt with DirectML support. Works on most GPUs, Windows only (fast).
            - `tensorrt` = GPU mode using vs-mlrt with TensorRT support. Requires an Nvidia RTX GPU (fastest).
        num_streams: Number of parallel GPU streams. Higher can be faster, but requires more VRAM. Does not effect the CPU backend.
        gpu_id: GPU index ID starting with 0 for the first compatible GPU. For example to switch between iGPU/dGPU. Does not effect the CPU backend.
        engine_folder: Optional path to the TensorRT engine storage location. By default engines are stored in `vs_colorfix/engines`. Only effects the TensorRT backend.
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
    if clip.format.bits_per_sample <= 8 or ref.format.bits_per_sample <= 8:
        warnings.simplefilter("always", UserWarning)
        warnings.warn("vs_colorfix.wavelet: Input clips have a low bit depth, which will cause banding. 16-bit input is recommended.", UserWarning, stacklevel=2)
    
    clip_format = clip.format
    num_planes  = clip.format.num_planes
    backend     = backend.lower()
    
    if clip_format.id != ref.format.id:
        raise ValueError("vs_colorfix.wavelet: Clip and ref must have the same format.")
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
    
    # clamp and shift uv if needed
    shift_uv = False
    if clip_format.sample_type == vs.FLOAT:
        clamp_expr = "x 0 max 1 min"
        if clip_format.color_family == vs.YUV:
            clamp_uv_expr = "x -0.5 max 0.5 min"
            if backend != "cpu":
                shift_uv = any(p in planes for p in (1, 2))
                clamp_uv_expr += " 0.5 +"  # shift uv to be within 0-1 for model input
            expr = [clamp_expr if 0 in planes else "", clamp_uv_expr if 1 in planes else "", clamp_uv_expr if 2 in planes else "",]
        else:
            expr = [clamp_expr if p in planes else "" for p in range(num_planes)]

        clip = _expression(clip, expr=expr)
        ref  = _expression(ref,  expr=expr)
    
    # color fix
    if backend == "cpu":
        clip = _wavelet_color_fix_atwt(clip, ref, wavelets=wavelets, planes=planes)
    else:
        clip = _wavelet_color_fix_vsmlrt(clip, ref, wavelets=wavelets, planes=planes, backend=backend, num_streams=num_streams, gpu_id=gpu_id, engine_folder=engine_folder)
    
    # undo uv shift if needed and return
    if shift_uv:
        return _expression(clip, expr=["", "x 0.5 -" if 1 in planes else "", "x 0.5 -" if 2 in planes else ""])
    return clip
