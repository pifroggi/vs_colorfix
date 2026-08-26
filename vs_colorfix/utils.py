
# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete and pifroggi on Discord

import os
import re
import shutil
import logging
import subprocess
import vapoursynth as vs
from pathlib import Path

core = vs.core


def expression(clips, expr, format=None):
    # optional plugin for slight speed boost
    if hasattr(core, "akarin"):
        return core.akarin.Expr(clips, expr, format=format)
    else:
        return core.std.Expr(clips, expr, format=format)


def box_blur(clip, planes=None, hradius=1, hpasses=1, vradius=1, vpasses=1):
    # optional plugin for slight speed boost
    if hasattr(core, "vszip"):
        return core.vszip.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)
    else:
        return core.std.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)


def make_diff(clipa, clipb, planes=None):
    # makes makediff work on 16-bit float
    if clipa.format.sample_type == vs.FLOAT and clipa.format.bits_per_sample == 16 and vs.__version__.release_major < 78:
        return expression([clipa, clipb], expr=["x y -" if i in planes else "" for i in range(clipa.format.num_planes)])
    else:
        return core.std.MakeDiff(clipa, clipb, planes=planes)


def merge_diff(clipa, clipb, planes=None):
    # makes mergediff work on 16-bit float
    if clipa.format.sample_type == vs.FLOAT and clipa.format.bits_per_sample == 16 and vs.__version__.release_major < 78:
        return expression([clipa, clipb], expr=["x y +" if i in planes else "" for i in range(clipa.format.num_planes)])
    else:
        return core.std.MergeDiff(clipa, clipb, planes=planes)


def compensate_wavelets(clip_format, wavelets):
    # compensate wavelet count for subsampled planes
    plane_wavelets = [wavelets] * clip_format.num_planes
    
    if clip_format.color_family == vs.YUV and clip_format.num_planes > 1:
        subsampling_shift = max(clip_format.subsampling_w, clip_format.subsampling_h)
        if subsampling_shift > 0:
            chroma_wavelets = max(1, wavelets - subsampling_shift)
            for p in range(1, clip_format.num_planes):
                plane_wavelets[p] = chroma_wavelets
    
    return plane_wavelets


def inference_groups(clips, planes, plane_wavelets):
    # group planes by wavelet count and size so interleaved inference only combines compatible planes
    groups = {}
    for p in sorted(planes):
        plane_clip = clips[p]
        key = (plane_wavelets[p], plane_clip.width, plane_clip.height)
        if key not in groups:
            groups[key] = []
        groups[key].append(p)
    
    return [(key[0], grouped_planes) for key, grouped_planes in groups.items()]


def pad(clip, pad_w, pad_h):
    # pad clip with repeated lines
    if pad_w:
        edge = core.std.CropAbs(clip, width=1, height=clip.height, left=clip.width - 1)
        clip = core.std.StackHorizontal([clip] + [edge] * pad_w)
    if pad_h:
        edge = core.std.CropAbs(clip, width=clip.width, height=1, top=clip.height - 1)
        clip = core.std.StackVertical([clip] + [edge] * pad_h)
    return clip


def get_builder(plugin_path, trt_version, cuda_major):
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
                raise RuntimeError("vs_colorfix: Internal Error: Regex failed to find the version.")

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
    raise FileNotFoundError(f"vs_colorfix: No compatible TensorRT engine builder found. Please install the python package 'tensorrt', or install trtexec. The required TensorRT version is {'.'.join(map(str, trt_version))}. The required CUDA version is {cuda_major}.\n{errors}")


def build_engine_trtexec(onnx_path, engine_path, engine_w, engine_h, num_planes, gpu_id, precision, opt_lvl, opt_tile, func_name, trt_version, trtexec_path):
    # build engine using trtexec, supports trt 10 and 11

    # settings
    opt_shapes = f"input:1x{num_planes*2}x{engine_h}x{engine_w}"
    min_shapes = f"input:1x{num_planes*2}x{engine_h}x{engine_w - 8}"  # avoid large engine sizes due to saving constants by making one dimension slightly dynamic
    io_formats = f"fp{precision}:chw" if trt_version[0] < 11 else "chw"
    cmd = [
        str(trtexec_path),
        *(["--stronglyTyped"] if trt_version[0] < 11 else []),
        *(["--tilingOptimizationLevel=3"] if opt_tile else []),
        "--skipInference",
        "--memPoolSize=workspace:4096",
        "--avgTiming=8",
        f"--builderOptimizationLevel={opt_lvl}",
        f"--inputIOFormats={io_formats}",
        f"--outputIOFormats={io_formats}",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={opt_shapes}",
        f"--device={gpu_id}",
    ]

    # build
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = (
            f"vs_colorfix{func_name}: Internal Error: trtexec failed while building the TensorRT engine.\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Return code: {e.returncode}\n"
        )
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e


def build_engine_python(onnx_path, engine_path, engine_w, engine_h, num_planes, gpu_id, opt_lvl, opt_tile, func_name, trt_package):
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
                    logging.critical(f"vs_colorfix{func_name}: Internal Error: TensorRT failed while building the TensorRT engine.\n=== TensorRT log ===\n{log}")
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
            raise RuntimeError(f"vs_colorfix{func_name}: Internal Error: TensorRT failed while parsing the ONNX model.\n{errors}")

        # settings
        opt_shapes = (1, num_planes * 2, engine_h, engine_w)                                                              # optShapes
        min_shapes = (1, num_planes * 2, engine_h, engine_w - 8)                                                          # minShapes
        network.get_input(0).allowed_formats = network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)  # IOFormats:chw
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 << 20)                                            # workspace
        config.avg_timing_iterations = 8                                                                                  # avgTiming
        config.builder_optimization_level = opt_lvl                                                                       # builderOptimizationLevel=5
        if opt_tile:
            config.tiling_optimization_level = trt.TilingOptimizationLevel.FULL                                           # tilingOptimizationLevel=3

        # build
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name, min_shapes, opt_shapes, opt_shapes)  # avoid large engine sizes due to saving constants by making one dimension slightly dynamic
        config.add_optimization_profile(profile)
        engine  = builder.build_serialized_network(network, config)
    finally:
        Device(cur_id).set_current()
        
    if engine is None:
        log = logger.get_log()
        msg = f"vs_colorfix{func_name}: Internal Error: TensorRT failed while building the TensorRT engine."
        if log:
            msg += f"\n=== TensorRT log ===\n{log}"
        raise RuntimeError(msg)
    
    # save engine
    with open(engine_path, "wb") as f:
        f.write(engine)


def get_engine(onnx_path, engine_dir, engine_w, engine_h, num_planes, precision, opt_lvl, opt_tile, gpu_id=0, force_rebuild=False, func_name="") -> str:
    # get path to tensorrt engine
    os.makedirs(engine_dir, exist_ok=True)  # create engine folder if needed
    model_name  = Path(onnx_path).stem
    engine_name = f"{model_name}_h{engine_h}_w{engine_w}_gpu{gpu_id}.engine"
    engine_path = os.path.join(engine_dir, engine_name)
    
    # check plugin version
    try:
        info = core.trt.Version()
    except Exception as e:
        raise RuntimeError(f"vs_colorfix{func_name}: TensorRT backend not installed. Please install the TensorRT dependencies with:\n'pip install -U vs_colorfix[tensorrt] --extra-index-url https://pypi.nvidia.com/'\nOr choose a different backend.") from e
    
    # if engine file exist, return it
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path
    
    # get plugin info
    plugin_path = os.path.dirname(info["path"].decode(errors="ignore"))
    trt_version = int(info["tensorrt_version"].decode(errors="ignore"))
    trt_version = [trt_version // 10000, (trt_version % 10000) // 100, trt_version % 100]
    cuda_major  = int(info["cuda_runtime_version"].decode(errors="ignore")) // 1000
    
    # build new engine
    logging.warning(f"vs_colorfix{func_name}: Building new TensorRT engine for width=%d, height=%d and precision=fp%d. This may take a few minutes.", engine_w, engine_h, precision)    
    builder_info = get_builder(plugin_path=plugin_path, trt_version=trt_version, cuda_major=cuda_major)
    if builder_info[0] == "python":
        build_engine_python(onnx_path=onnx_path, engine_path=engine_path, engine_w=engine_w, engine_h=engine_h, num_planes=num_planes, gpu_id=gpu_id, opt_lvl=opt_lvl, opt_tile=opt_tile, func_name=func_name, trt_package=builder_info[1])
    elif builder_info[0] == "trtexec":
        build_engine_trtexec(onnx_path=onnx_path, engine_path=engine_path, engine_w=engine_w, engine_h=engine_h, num_planes=num_planes, gpu_id=gpu_id, opt_lvl=opt_lvl, opt_tile=opt_tile, func_name=func_name, precision=precision, trt_version=trt_version, trtexec_path=builder_info[1])
    else:
        raise RuntimeError(f"vs_colorfix{func_name}: Internal Error: Unknown TensorRT engine builder: {builder_info[0]}")
    logging.warning(f"vs_colorfix{func_name}: Engine building complete.")
    return engine_path


def tensorrt_inference(input_clips, onnx_path, engine_dir, clip_w, clip_h, num_planes, precision, opt_lvl, opt_tile, num_streams=1, gpu_id=0, force_rebuild=False, func_name=""):
    engine_path = get_engine(onnx_path=onnx_path, engine_dir=engine_dir, engine_w=clip_w, engine_h=clip_h, num_planes=num_planes, precision=precision, gpu_id=gpu_id, opt_lvl=opt_lvl, opt_tile=opt_tile, func_name=func_name, force_rebuild=force_rebuild)
    model_args  = dict(engine_path=engine_path, num_streams=num_streams, device_id=gpu_id)
    
    # try inference, rebuild engine if it fails
    try:
        out = core.trt.Model(input_clips, **model_args)
    except vs.Error as e:
        err_msg = str(e).lower()
        serialization_keywords = ("serialize", "serialization", "deserialize", "deserialization")
        if any(k in err_msg for k in serialization_keywords) and not force_rebuild:
            logging.warning(f"vs_colorfix{func_name}: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
            model_args["engine_path"] = get_engine(onnx_path=onnx_path, engine_dir=engine_dir, engine_w=clip_w, engine_h=clip_h, num_planes=num_planes, precision=precision, gpu_id=gpu_id, opt_lvl=opt_lvl, opt_tile=opt_tile, func_name=func_name, force_rebuild=True)
            out = core.trt.Model(input_clips, **model_args)
        else:
            raise
    return out
