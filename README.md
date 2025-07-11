# Color Fix functions for Vapoursynth

For example for transfering colors from one source to another, or fixing color shift from AI upscaling/restoration models. Also known as Color Transfer or Color Matching sometimes.  
Example fixing colors after upscaling a DVD: https://imgsli.com/MjM5NzM5/0/2

### Requirements
* [pytorch with cuda](https://pytorch.org/) *(optional, only for Wavelet Color Fix)*
* `pip install numpy` *(optional, only for Wavelet Color Fix)*
* [vszip](https://github.com/dnjulek/vapoursynth-zip) *(optional, speed boost for Average Color Fix)*

### Setup
Put the `vs_colorfix.py` file into your vapoursynth scripts folder.  
Or install via pip: `pip install -U git+https://github.com/pifroggi/vs_colorfix.git`

<br />

## Average Color Fix
Correct for color shift by matching the average color of a clip to that of a reference clip. This is a very fast way to transfer the colors from one clip to another that has the same or close to the same content, but different colors. Idea from [chaiNNer](https://github.com/chaiNNer-org/chaiNNer).

```python
import vs_colorfix
clip = vs_colorfix.average(clip, ref, radius=10, planes=[0, 1, 2], fast=False)
```

__*`clip`*__  
Clip where the color fix will be applied to.  
Any format, but recommended is a bit depth higher than 8 to avoid banding.

__*`ref`*__  
Reference clip where the colors are taken from.  
Any format, but recommended is a bit depth higher than 8 to avoid banding.

__*`radius`*__  
Higher means a more global color match and wider bloom/bleed.  
Lower means a more local color match and smaller bloom/bleed. Too low and the reference clip will become visible.  
Test values 5 and 30 and this will become more clear.

__*`planes`* (optional)__  
Which planes to color fix. Any unmentioned planes will simply be copied.  
If nothing is set, all planes will be color fixed.

 __*`fast`* (optional)__  
Does the averaging via a downscale instead of a blur, which is much faster, but will produce faint blocky artifacts.  
I found it useful for radius > 30 where artifacts are no longer noticable, or to fix something like a prefilter clip.

## Wavelet Color Fix
Correct for color shift by first separating a clip into different frequencies (wavelets), then matching the average color to that of a reference clip. This works similarly to the Average Color Fix, but produces better results at the cost of more computation. Both clips must have close to the same content. The Wavelet Color Fix functions are from [sd-webui-stablesr](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py).  

```python
import vs_colorfix
clip = vs_colorfix.wavelet(clip, ref, wavelets=5, planes=[0, 1, 2], device="cuda")
```

__*`clip`*__  
Clip where the color fix will be applied to.  
Must be in YUV444PS, YUV444PH, RGBS, RGBH, GRAYS, or GRAYH format.

__*`ref`*__  
Reference clip where the colors are taken from.  
Must be in YUV444PS, YUV444PH, RGBS, RGBH, GRAYS, or GRAYH format.

__*`wavelets`*__  
Number of wavelets, 5 seems to work best in most cases.  
Higher means a more global color match and wider bloom/bleed.  
Lower means a more local color match and smaller bloom/bleed. Too low and the reference clip will become visible.  
Test values 3 and 10 and this will become more clear.

__*`planes`* (optional)__  
Which planes to color fix. Any unmentioned planes will simply be copied.  
If nothing is set, all planes will be color fixed.

__*`device`* (optional)__  
Device can be "cpu" to use the CPU, or "cuda" to use an Nvidia GPU.  
YUV444PH, RGBH, or GRAYH format will additionally __double speed__ if the GPU supports fp16.

<br />

## Tips & Troubleshooting
> [!TIP]
> * If your clips are not sufficiently aligned or synchronized, try this: https://github.com/pifroggi/vs_align
> * To replicate chaiNNers Average Color Fix, you can convert % to radius: `radius = (100/percentage-1)/2`  
>   ChaiNNer works like fast=True does here, but it is recommended to leave it off for better results.
