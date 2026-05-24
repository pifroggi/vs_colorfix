
# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete and pifroggi on Discord

# Average Color Fix idea from chaiNNer https://github.com/chaiNNer-org/chaiNNer

import vapoursynth as vs
import warnings

core = vs.core


def _expression(clips, expr, format=None):
    # optional plugin for slight speed boost
    if hasattr(core, "akarin"):
        return core.akarin.Expr(clips, expr, format=format)
    else:
        return core.std.Expr(clips, expr, format=format)


def _box_blur(clip, planes=None, hradius=1, hpasses=1, vradius=1, vpasses=1):
    # optional plugin for slight speed boost
    if hasattr(core, "vszip"):
        return core.vszip.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)
    else:
        return core.std.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)


def _make_diff(clipa, clipb, planes=None):
    # makes makediff work on 16-bit float
    if clipa.format.sample_type == vs.FLOAT and clipa.format.bits_per_sample == 16:
        return _expression([clipa, clipb], expr=["x y -" if i in planes else "" for i in range(clipa.format.num_planes)])
    else:
        return core.std.MakeDiff(clipa, clipb, planes=planes)


def _merge_diff(clipa, clipb, planes=None):
    # makes mergediff work on 16-bit float
    if clipa.format.sample_type == vs.FLOAT and clipa.format.bits_per_sample == 16:
        return _expression([clipa, clipb], expr=["x y +" if i in planes else "" for i in range(clipa.format.num_planes)])
    else:
        return core.std.MergeDiff(clipa, clipb, planes=planes)


def average_color_fix(clip, ref, radius=10, planes=None, fast=False):
    """Fixes color shift based on a reference clip. A very fast way to transfer the colors from one clip to another. For large color differences, `wavelet()` is more accurate. Both clips must have close to the same content.

    Args:
        clip: Clip where the color fix will be applied to. Recommended is a bit depth higher than 8 to avoid banding.
        ref: Reference clip where the colors are taken from. Recommended is a bit depth higher than 8 to avoid banding.
        radius: Higher means a more global color match and wider bloom/bleed. Lower means a more local color match and smaller bloom/bleed. 
            Too low and the reference clip will become visible. Test values 5 and 30 and this will become more clear.
        planes: Which planes to color fix. Any unmentioned planes will simply be copied. None means all planes will be color fixed.
        fast: Does the averaging via a downscale instead of a blur, which is much faster, but will produce faint blocky artifacts. 
            I found it useful for radius > 60 where artifacts are often no longer noticable, or to fix something like a prefilter clip.
    """
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_colorfix.average: Clip must be a vapoursynth clip.")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("vs_colorfix.average: Ref must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_colorfix.average: Clip must have constant format and dimensions.")
    if ref.format.id  == vs.PresetVideoFormat.NONE or  ref.width == 0 or  ref.height == 0:
        raise TypeError("vs_colorfix.average: Ref must have constant format and dimensions.")
    if clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample == 16:
        raise ValueError("vs_colorfix.average: 16-bit float formats is not supported.")  # vszip allows it, but creates artifacts https://github.com/dnjulek/vapoursynth-zip/issues/20
    if clip.format.id != ref.format.id:
        raise ValueError("vs_colorfix.average: Clip and ref must have the same format. 16-bit input is recommended to avoid banding.")
    if not isinstance(fast, bool):
        raise TypeError("vs_colorfix.average: Fast must be either True or False.")
    if not isinstance(radius, int) or isinstance(radius, bool):
        raise TypeError("vs_colorfix.average: Radius must be an integer.")
    if not fast and not (1 <= radius <= 10000):
        raise ValueError("vs_colorfix.average: Radius must be in the range 1-10000.")
    fast_max_radius = (min(clip.width, clip.height) - 1) // 2
    if fast and not (1 <= radius <= fast_max_radius):
        raise ValueError(f"vs_colorfix.average: Radius must be between 1 and {fast_max_radius} for the current input dimensions when fast=True.")
    if clip.format.bits_per_sample <= 8 or ref.format.bits_per_sample <= 8:
        warnings.simplefilter("always", UserWarning)
        warnings.warn("vs_colorfix.average: Input clips have a low bit depth, which will cause banding. 16-bit input is recommended.", UserWarning, stacklevel=2)
    
    num_planes = clip.format.num_planes
    if planes is None:
        planes = list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if num_planes == 1:
        planes = [0]
    planes = set(planes)
    if not planes <= set(range(num_planes)):
        raise ValueError("vs_colorfix.average: Invalid plane index specified.")

    # downscale both clips, calculate difference (faster but faint blocky artifacts)
    if fast:
        radius = radius * 2 + 1
        processed_clips = [None] * num_planes
        if 0 in planes:
            clip_plane = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
            ref_plane = core.std.ShufflePlanes(ref, planes=0, colorfamily=vs.GRAY)
            downscaled_clip_plane = core.resize.Bilinear(clip_plane, width=clip.width // radius, height=clip.height // radius)
            downscaled_ref_plane = core.resize.Bilinear(ref_plane, width=clip.width // radius, height=clip.height // radius)
            diff_plane = _make_diff(downscaled_ref_plane, downscaled_clip_plane, planes=[0])
            processed_clips[0] = core.resize.Bilinear(diff_plane, width=clip.width, height=clip.height)
        else:
            processed_clips[0] = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
        if 1 in planes:
            clip_plane = core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
            ref_plane = core.std.ShufflePlanes(ref, planes=1, colorfamily=vs.GRAY)
            downscaled_clip_plane = core.resize.Bilinear(clip_plane, width=clip.width // radius, height=clip.height // radius)
            downscaled_ref_plane = core.resize.Bilinear(ref_plane, width=clip.width // radius, height=clip.height // radius)
            diff_plane = _make_diff(downscaled_ref_plane, downscaled_clip_plane, planes=[0])
            processed_clips[1] = core.resize.Bilinear(diff_plane, width=clip_plane.width, height=clip_plane.height)
        elif num_planes > 1:
            processed_clips[1] = core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
        if 2 in planes:
            clip_plane = core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)
            ref_plane = core.std.ShufflePlanes(ref, planes=2, colorfamily=vs.GRAY)
            downscaled_clip_plane = core.resize.Bilinear(clip_plane, width=clip.width // radius, height=clip.height // radius)
            downscaled_ref_plane = core.resize.Bilinear(ref_plane, width=clip.width // radius, height=clip.height // radius)
            diff_plane = _make_diff(downscaled_ref_plane, downscaled_clip_plane, planes=[0])
            processed_clips[2] = core.resize.Bilinear(diff_plane, width=clip_plane.width, height=clip_plane.height)
        elif num_planes > 2:
            processed_clips[2] = core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)
        diff_clip = core.std.ShufflePlanes(clips=processed_clips, planes=[0] * num_planes, colorfamily=clip.format.color_family)

    # blur both clips, calculate difference (better quality but slower)
    else:
        if ref.width != clip.width or ref.height != clip.height:
            ref = core.resize.Bilinear(ref, width=clip.width, height=clip.height)
        chroma_hradius = radius // (1 << clip.format.subsampling_w) if clip.format.subsampling_w else radius
        chroma_vradius = radius // (1 << clip.format.subsampling_h) if clip.format.subsampling_h else radius
        blurred_clip = clip
        blurred_ref = ref
        if 0 in planes:
            blurred_clip = _box_blur(blurred_clip, hradius=radius, hpasses=4, vradius=radius, vpasses=4, planes=[0])
            blurred_ref = _box_blur(blurred_ref, hradius=radius, hpasses=4, vradius=radius, vpasses=4, planes=[0])
        if 1 in planes:
            blurred_clip = _box_blur(blurred_clip, hradius=chroma_hradius, hpasses=4, vradius=chroma_vradius, vpasses=4, planes=[1])
            blurred_ref = _box_blur(blurred_ref, hradius=chroma_hradius, hpasses=4, vradius=chroma_vradius, vpasses=4, planes=[1])
        if 2 in planes:
            blurred_clip = _box_blur(blurred_clip, hradius=chroma_hradius, hpasses=4, vradius=chroma_vradius, vpasses=4, planes=[2])
            blurred_ref = _box_blur(blurred_ref, hradius=chroma_hradius, hpasses=4, vradius=chroma_vradius, vpasses=4, planes=[2])
        diff_clip = _make_diff(blurred_ref, blurred_clip, planes=planes)

    # add difference to the original
    return _merge_diff(clip, diff_clip, planes=planes)
