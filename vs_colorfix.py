
# Wavelet Color Fix from "sd-webui-stablesr" https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py
# Average Color Fix idea from "chaiNNer" https://github.com/chaiNNer-org/chaiNNer

# Script by pifroggi https://github.com/pifroggi/vs_colorfix
# or tepete on the "Enhance Everything!" Discord Server

import vapoursynth as vs
import numpy as np
import torch
import torch.nn.functional as F
core = vs.core

#frame conversion functions
def array_to_frame(img: np.ndarray, frame: vs.VideoFrame):
    #directly copy the float32 data
    for p in range(3):
        pls = frame[p]
        frame_arr = np.asarray(pls)
        np.copyto(frame_arr, img[:, :, p])

def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    #directly return the numpy array for float32 data
    return np.dstack([np.asarray(frame[p]) for p in range(frame.format.num_planes)])



#functions for wavelet processing
def wavelet_blur(image: torch.Tensor, radius: int):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: torch.Tensor, levels):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq

def wavelet_reconstruction(content_feat: torch.Tensor, style_feat: torch.Tensor, levels):
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat, levels=levels)
    del content_low_freq
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat, levels=levels)
    del style_high_freq
    return content_high_freq + style_low_freq

def wavelet_color_fix(target_frame: np.ndarray, source_frame: np.ndarray, levels):
    to_tensor = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)
    target_tensor = to_tensor(target_frame)
    source_tensor = to_tensor(source_frame)
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor, levels=levels)
    to_image = lambda x: x.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    result_image = to_image(result_tensor)
    return result_image



#wavelet color fix
def wavelet(clip, reference_clip, wavelets=5):
    #check if both clips are in float32
    required_formats = [vs.RGBS, vs.YUV444PS]
    if clip.format.id not in required_formats or reference_clip.format.id not in required_formats:
        raise ValueError("Input clips must be in RGBS or YUV444PS format.")

    #resize reference_clip to match the dimensions of clip
    reference_clip = reference_clip.resize.Bicubic(width=clip.width, height=clip.height)

    def do_wavelet_color_fix(n, f, levels=wavelets):
        fout = f[1].copy()
        target_frame = frame_to_array(f[1])
        source_frame = frame_to_array(f[0])

        #apply wavelet color fix on numpy arrays
        result_np = wavelet_color_fix(target_frame, source_frame, levels=levels)

        #convert back to vapoursynth frame
        array_to_frame(result_np, fout)

        return fout

    modified_clip = core.std.ModifyFrame(clip=clip, clips=[reference_clip, clip], selector=do_wavelet_color_fix)

    return modified_clip



def average(clip, reference_clip, blur_radius):

    #check if clips have a low bit depth
    low_bit_depth_formats = [
        vs.RGB24,
        vs.YUV420P8,
        vs.YUV422P8,
        vs.YUV444P8,
        vs.YUV410P8,
        vs.YUV411P8,
        vs.YUV440P8,
        vs.GRAY8
    ]
    if clip.format.id in low_bit_depth_formats or reference_clip.format.id in low_bit_depth_formats:
        raise ValueError("Input clips must have a bit depth higher than 8 to avoid banding.")

    #resize reference to clip size
    reference_clip = core.resize.Bilinear(reference_clip, width=clip.width, height=clip.height)
    
    #blur both clips
    blurred_reference = core.std.BoxBlur(reference_clip, hradius=blur_radius, hpasses=4, vradius=blur_radius, vpasses=4)
    blurred_clip = core.std.BoxBlur(clip, hradius=blur_radius, hpasses=4, vradius=blur_radius, vpasses=4)

    #calculate difference
    diff_clip = core.std.MakeDiff(blurred_reference, blurred_clip)

    #add difference to original
    fixed_clip = core.std.MergeDiff(clip, diff_clip)

    return fixed_clip