
# Wavelet Color Fix from "sd-webui-stablesr" https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py
# Average Color Fix idea from "chaiNNer" https://github.com/chaiNNer-org/chaiNNer

# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete on the "Enhance Everything!" Discord Server

import vapoursynth as vs
import numpy as np
import torch
import torch.nn.functional as F
import warnings

core = vs.core

def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame):
    array = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    for p in range(array.shape[2]):
        np.copyto(np.asarray(frame[p]), array[:, :, p])

def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    planes = [torch.as_tensor(np.array(frame[p], copy=True), dtype=torch.float32) for p in range(frame.format.num_planes)]
    return torch.stack(planes, dim=0).unsqueeze(0)

def wavelet_blur(image: torch.Tensor, radius: int):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(image.size(1), 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    return F.conv2d(image, kernel, groups=image.size(1), dilation=radius)

def wavelet_decomposition(image: torch.Tensor, levels: int):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq

def wavelet_reconstruction(content_feat: torch.Tensor, style_feat: torch.Tensor, levels: int):
    content_high_freq, _ = wavelet_decomposition(content_feat, levels=levels)
    _, style_low_freq = wavelet_decomposition(style_feat, levels=levels)
    return content_high_freq + style_low_freq

def wavelet(clip, ref, wavelets=5, planes=None):
    supported_formats = [vs.RGBS, vs.YUV444PS, vs.GRAYS]
    if clip.format.id not in supported_formats or ref.format.id not in supported_formats:
        raise ValueError("Input clips must be in RGBS, YUV444PS, or GRAYS format.")
    if clip.format.id != ref.format.id:
        raise ValueError("Clip and ref must have the same format.")
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if clip.format.num_planes == 1:
        planes = [0]
    if ref.width != clip.width or ref.height != clip.height:
        ref = core.resize.Bicubic(ref, width=clip.width, height=clip.height)
    UV = clip.format.id == vs.YUV444PS and any(p > 0 for p in planes)

    def wavelet_color_fix(n, f, levels=wavelets):
        fout = f[1].copy()

        target_tensor = frame_to_tensor(f[1])
        source_tensor = frame_to_tensor(f[0])

        #normalize UV
        if UV:
            target_tensor[:, 1:, :, :] += 0.5
            source_tensor[:, 1:, :, :] += 0.5

        #select planes
        target_selected = target_tensor[:, planes, :, :]
        source_selected = source_tensor[:, planes, :, :]

        #colorfix
        result_tensor = wavelet_reconstruction(target_selected, source_selected, levels=levels)
        result_tensor = result_tensor.clamp(0, 1)

        #recombine with unprocessed planes
        combined_tensor = target_tensor.clone()
        combined_tensor[:, planes, :, :] = result_tensor

        #unnormalize UV
        if UV:
            combined_tensor[:, 1:, :, :] -= 0.5

        tensor_to_frame(combined_tensor, fout)
        return fout

    return core.std.ModifyFrame(clip=clip, clips=[ref, clip], selector=wavelet_color_fix)







def average(clip, ref, radius=10, planes=None, fast=False):
    if clip.format.id != ref.format.id:
        raise ValueError("Clip and ref must have the same format.")
    if clip.format.bits_per_sample <= 8 or ref.format.bits_per_sample <= 8:
        warnings.warn("Input clips have a low bit depth, which will cause banding. 16 bit input is recommended.", UserWarning)
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if clip.format.num_planes == 1:
        planes = [0]
    
    #downscale both clips, calculate difference (faster but faint blocky artifacts)
    if fast:
        processed_clips = []
        for i in range(clip.format.num_planes):
            if i in planes:
                clip_plane = core.std.ShufflePlanes(clip, planes=i, colorfamily=vs.GRAY)
                ref_plane  = core.std.ShufflePlanes(ref,  planes=i, colorfamily=vs.GRAY)
                downscaled_clip_plane = core.resize.Bilinear(clip_plane, width=clip.width // radius, height=clip.height // radius)
                downscaled_ref_plane  = core.resize.Bilinear(ref_plane,  width=clip.width // radius, height=clip.height // radius)
                diff_plane = core.std.MakeDiff(downscaled_ref_plane, downscaled_clip_plane, planes=0)
                diff_plane = core.resize.Bilinear(diff_plane, width=clip.width, height=clip.height)
            else:
                diff_plane = core.std.ShufflePlanes(clip, planes=i, colorfamily=vs.GRAY)
            processed_clips.append(diff_plane)
        diff_clip = core.std.ShufflePlanes(clips=processed_clips, planes=[0]*clip.format.num_planes, colorfamily=clip.format.color_family)
    
    #blur both clips, calculate difference (better quality but slower)
    else:
        if ref.width != clip.width or ref.height != clip.height:
            ref = core.resize.Bilinear(ref, width=clip.width, height=clip.height)
        blurred_clip = core.std.BoxBlur(clip, hradius=radius, hpasses=4, vradius=radius, vpasses=4, planes=planes)
        blurred_ref  = core.std.BoxBlur(ref,  hradius=radius, hpasses=4, vradius=radius, vpasses=4, planes=planes)
        diff_clip = core.std.MakeDiff(blurred_ref, blurred_clip, planes=planes)

    #add difference to the original
    return core.std.MergeDiff(clip, diff_clip, planes=planes)
