
# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete and pifroggi on Discord

# Wavelet Color Fix idea from sd-webui-stablesr https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py

import os
import re
import sys
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


def _get_builder(plugin_path, trt_version, cuda_major):
    # finds compatible tensorrt engine builders
    exe_name = "trtexec.exe" if os.name == "nt" else "trtexec"
    builders = []
    errors   = []
    
    # check for python tensorrt
    try:
        import tensorrt
        package_version = list(map(int, tensorrt.__version__.split(".")[:3]))
        if package_version == trt_version:
            builders.append(["python", tensorrt])
        else:
            errors.append(f"Python TensorRT: Wrong version {'.'.join(map(str, package_version))}")
    except ImportError:
        errors.append("Python TensorRT: Not found.")
    except Exception:
        errors.append("Python TensorRT: Found but failed to check version.")
    
    # check for bundled trtexec
    bundled_trtexec = Path(plugin_path) / "vsmlrt-cuda" / exe_name
    if bundled_trtexec.is_file() and os.access(str(bundled_trtexec), os.X_OK):
        builders.append(["trtexec", bundled_trtexec])
    else:
        errors.append(f"Bundled trtexec: Not found.")

    # check for system trtexec
    system_trtexec = shutil.which("trtexec")
    if system_trtexec is not None:
        try:
            trtexec_path = Path(system_trtexec)
            help_output  = subprocess.run([str(trtexec_path), "--help"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
            help_output  = f"{help_output.stdout}\n{help_output.stderr}"
            
            trtexec_version = None
            trtexec_version = re.search(r"\[TensorRT v(\d+)\]", help_output)
            if trtexec_version is None:
                raise RuntimeError("vs_colorfix.wavelet: Internal Error: Regex failed to find the version.")

            trtexec_version = int(trtexec_version.group(1))
            trtexec_version = [trtexec_version // 10000, (trtexec_version % 10000) // 100, trtexec_version % 100]
            if trtexec_version == trt_version:
                builders.append(["trtexec", trtexec_path])
            else:
                errors.append(f"System trtexec: Wrong version {'.'.join(map(str, trtexec_version))}")
        except Exception:
            errors.append("System trtexec: Found but failed to check version.")
    else:
        errors.append("System trtexec: Not found.")
    
    # return first compatible builder
    if builders:
        return builders[0]
    
    errors = "\n".join(f"{builder}" for builder in errors)
    raise FileNotFoundError(f"vs_colorfix.wavelet: No compatible TensorRT engine builder found. Please install the python packages 'tensorrt', or install trtexec. The required TensorRT version is {'.'.join(map(str, trt_version))}. The required CUDA version is {cuda_major}.\n{errors}")


def _build_engine_trtexec(onnx_path, engine_path, engine_w, engine_h, num_planes, gpu_id, precision, trt_version, trtexec_path):
    # build engine using trtexec, supports trt 10 and 11

    # settings
    opt_shapes = f"input:1x{num_planes*2}x{engine_h}x{engine_w}"
    io_formats = f"fp{precision}:chw" if trt_version[0] < 11 else "chw"
    cmd = [
        str(trtexec_path),
        *(["--stronglyTyped"] if trt_version[0] < 11 else []),
        "--skipInference",
        "--memPoolSize=workspace:4096",
        "--builderOptimizationLevel=5",
        "--tilingOptimizationLevel=3",
        f"--inputIOFormats={io_formats}",
        f"--outputIOFormats={io_formats}",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--optShapes={opt_shapes}",
        f"--device={gpu_id}",
    ]

    # build
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = (
            "vs_colorfix.wavelet: Internal Error: trtexec failed while building the TensorRT engine.\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Return code: {e.returncode}\n"
        )
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e


def _build_engine_python(onnx_path, engine_path, engine_w, engine_h, num_planes, gpu_id, trt_package):
    # build engine using tensorrt python package, supports only trt 11 because of vapoursynth-mlrt-trt
    from cuda.core import Device
    trt = trt_package

    # custom logger for errors
    class _TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages = []
            self.fatal    = False
        def log(self, severity, msg):
            if severity <= trt.Logger.WARNING:
                self.messages.append((severity, msg))
                if self.fatal:
                    logging.critical(f"  [{severity}] {msg}")
                elif severity == trt.Logger.INTERNAL_ERROR:  # print fatal errors immediately because python may not get control back
                    self.fatal = True
                    log = "\n".join(f"  [{log_severity}] {log_msg}" for log_severity, log_msg in self.messages)
                    logging.critical(f"vs_colorfix.wavelet: Internal Error: TensorRT failed while building the TensorRT engine.\n=== TensorRT log ===\n{log}")
        def get_log(self):
            return "\n".join(f"  [{severity}] {msg}" for severity, msg in self.messages)

    # initialize trt and load model
    cur_id = Device().device_id
    try:
        Device(gpu_id).set_current()
        logger  = _TrtLogger()
        builder = trt.Builder(logger)
        network = builder.create_network()
        config  = builder.create_builder_config()
        parser  = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "\n".join(f"  {parser.get_error(i)}" for i in range(parser.num_errors))
            raise RuntimeError(f"vs_colorfix.wavelet: Internal Error: TensorRT failed while parsing the ONNX model.\n{errors}")

        # settings
        opt_shapes = (1, num_planes * 2, engine_h, engine_w)                                                              # optShapes
        network.get_input(0).allowed_formats = network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)  # IOFormats:chw
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 << 20)                                            # workspace
        config.builder_optimization_level = 5                                                                             # builderOptimizationLevel=5
        config.tiling_optimization_level  = trt.TilingOptimizationLevel.FULL                                              # tilingOptimizationLevel=3

        # build
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name, opt_shapes, opt_shapes, opt_shapes)
        config.add_optimization_profile(profile)
        engine  = builder.build_serialized_network(network, config)
    finally:
        Device(cur_id).set_current()
        
    if engine is None:
        log = logger.get_log()
        msg = "vs_colorfix.wavelet: Internal Error: TensorRT failed while building the TensorRT engine."
        if log:
            msg += f"\n=== TensorRT log ===\n{log}"
        raise RuntimeError(msg)
    
    # save engine
    with open(engine_path, "wb") as f:
        f.write(engine)


def _get_engine(onnx_path, engine_dir, engine_w, engine_h, num_planes, precision, gpu_id=0, force_rebuild=False) -> str:
    # get path to tensorrt engine
    os.makedirs(engine_dir, exist_ok=True)  # create engine folder if needed
    model_name  = Path(onnx_path).stem
    engine_name = f"{model_name}_h{engine_h}_w{engine_w}_gpu{gpu_id}.engine"
    engine_path = os.path.join(engine_dir, engine_name)
    
    # check plugin version
    try:
        info = core.trt.Version()
    except Exception as e:
        raise RuntimeError("vs_colorfix.wavelet: TensorRT backend not installed. Please install the TensorRT dependencies with:\n'pip install -U vs_colorfix[tensorrt] --extra-index-url https://pypi.nvidia.com/'\nOr choose a different backend.") from e
    
    # if engine file exist, return it
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path
    
    # get plugin info
    plugin_path = os.path.dirname(info["path"].decode(errors="ignore"))
    trt_version = int(info["tensorrt_version"].decode(errors="ignore"))
    trt_version = [trt_version // 10000, (trt_version % 10000) // 100, trt_version % 100]
    cuda_major  = int(info["cuda_runtime_version"].decode(errors="ignore")) // 1000
    
    # build new engine
    logging.warning("vs_colorfix.wavelet: Building new TensorRT engine for width=%d, height=%d and precision=fp%d. This may take a few minutes.", engine_w, engine_h, precision)    
    builder_info = _get_builder(plugin_path=plugin_path, trt_version=trt_version, cuda_major=cuda_major)
    if builder_info[0] == "python":
        _build_engine_python(onnx_path=onnx_path, engine_path=engine_path, engine_w=engine_w, engine_h=engine_h, num_planes=num_planes, gpu_id=gpu_id, trt_package=builder_info[1])
    elif builder_info[0] == "trtexec":
        _build_engine_trtexec(onnx_path=onnx_path, engine_path=engine_path, engine_w=engine_w, engine_h=engine_h, num_planes=num_planes, gpu_id=gpu_id, precision=precision, trt_version=trt_version, trtexec_path=builder_info[1])
    else:
        raise RuntimeError(f"vs_colorfix.wavelet: Internal Error: Unknown TensorRT engine builder: {builder_info[0]}")
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
    """Fixes color shift based on a reference clip. Works similarly to `average()`, but more accurate when color differences are large, at the cost of more computation.

    Args:
        clip: Clip where the color fix will be applied to.
        ref: Reference clip where the colors are taken from. Should match the base clip somewhat. Compression, grain, or lower resolution are all okay.
        wavelets: Number of wavelets, around 4 seems to work best in most cases. Higher means a more global color match and wider bloom/bleed. Lower means a more 
            local color match and smaller bloom/bleed. Lower is also faster. Too low and the reference clip will become visible. Test values 3 and 8 and this will become more clear.
        planes: Which planes to color fix. Any unmentioned planes will simply be copied. None means all planes will be color fixed.
        backend: The backend used to run the color fix. **16-bit float input is always much faster on GPU, but not supported by older GPUs.**
            - `cpu` = CPU mode using the Vapoursynth-ATWT plugin (slow).
            - `ncnn` = GPU mode using vs-mlrt with NCNN support. Works on almost any GPU, even MAC (fast).
            - `directml` = GPU mode using vs-mlrt with DirectML support. Works on most GPUs, but Windows only (fast).
            - `tensorrt` = GPU mode using vs-mlrt with TensorRT support. Requires an Nvidia RTX GPU (very fast).
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
    if clip.format.bits_per_sample <= 8 or ref.format.bits_per_sample <= 8:
        warnings.warn("vs_colorfix.wavelet: Input clips have a low bit depth, which will cause banding. 16-bit input is recommended.", UserWarning, stacklevel=2)
    
    clip_format = clip.format
    num_planes  = clip.format.num_planes
    backend     = backend.lower()
    
    if backend in ["directml", "dml"] and sys.platform != "win32":
        raise RuntimeError("vs_colorfix.wavelet: The DirectML backend is only available on Windows.")
    if backend in ["tensorrt", "trt"] and sys.platform not in ("win32", "linux"):
        raise RuntimeError("vs_colorfix.wavelet: The TensorRT backend is only available on Windows and Linux.")
    if backend in ["cpu"] and not hasattr(core, "atwt"):
        raise RuntimeError("vs_colorfix.wavelet: Please install the plugin 'Vapoursynth-ATWT' to use the CPU backend.")
    
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
