# Color Fix for VapourSynth

For example for fixing color shift from AI upscaling/restoration models, or transferring color grading from an old release to a remaster. Also known as Color Transfer or Color Matching. See this collection of [Comparisons](https://slow.pics/c/abXjnKn3).

<br />

<p align="center">
  <a href="https://slow.pics/c/abXjnKn3">
    <img src="https://raw.githubusercontent.com/pifroggi/vs_colorfix/refs/heads/main/README_img.png" width="600" />
  </a>
</p>

## Installation

### Nvidia
```
pip install -U vs_colorfix[tensorrt] --extra-index-url https://pypi.nvidia.com/
```
To enable the Wavelet Color Fix CPU backend, install the [ATWT](https://github.com/yuygfgg/Vapoursynth-ATWT) plugin. *(optional)*  
### Others
```
pip install -U vs_colorfix
```
To enable the Wavelet Color Fix CPU backend, install the [ATWT](https://github.com/yuygfgg/Vapoursynth-ATWT) plugin. *(optional)*  

<br />

> [!TIP]
> For VapourSynth R73 and older, follow the [manual installation steps](https://github.com/pifroggi/vs_colorfix/wiki/Manual-Installation).

<br />


## Average Color Fix
Fixes color shift by matching the average color of a clip to a reference clip. A very fast way to transfer colors from one clip to another.

```python
import vs_colorfix
clip = vs_colorfix.average(clip, ref, radius=10, planes=[0, 1, 2], fast=False)
```

__*`clip`*__  
Base clip where the colors will be applied to.  
Recommended higher than 8-bit to avoid banding.

__*`ref`*__  
Reference clip where the colors are taken from.  
Check the [comparisons](https://slow.pics/c/abXjnKn3) to get an idea how close it should match the base clip.

__*`radius`*__  
Higher means a more global color match and wider bloom/bleed.  
Lower means a more local color match and smaller bloom/bleed. Too low and the reference clip will become visible.  
Test values 5 and 30 and this will become more clear.

__*`planes`* (optional)__  
Which planes to color fix. Any unmentioned planes will simply be copied.  
If not set, all planes will be color-fixed.

 __*`fast`* (optional)__  
Does the averaging via a downscale instead of a blur, which is much faster, but will produce faint blocky artifacts.  
Useful for very large radii where artifacts are no longer noticeable, or to fix something like a prefilter clip.

> [!TIP]
> * If your clips are not sufficiently aligned or synchronized, use [vs_align](https://github.com/pifroggi/vs_align) to align them first.
> * To replicate [chaiNNers](https://github.com/chaiNNer-org/chaiNNer) Average Color Fix, convert percentage to radius: `radius = (100/percentage-1)/2`  
>   ChaiNNer works like fast=True does here, but it is recommended to leave it off for better results.

<br />

## Wavelet Color Fix
Fixes color shift by converting into wavelets, then matching the average color of a clip to a reference clip. Works similarly to the Average Color Fix, but more accurate for larger color differences, at the cost of more computation.

```python
import vs_colorfix
clip = vs_colorfix.wavelet(clip, ref, wavelets=4, planes=[0, 1, 2], backend="ncnn", num_streams=2, gpu_id=0, engine_folder=None)
```

__*`clip`*__  
Base clip where the colors will be applied to.  
Recommended higher than 8-bit to avoid banding.

__*`ref`*__  
Reference clip where the colors are taken from.  
Check the [comparisons](https://slow.pics/c/abXjnKn3) to get an idea how close it should match the base clip.

__*`wavelets`*__  
Number of wavelets in the 1-10 range. Around 4 seems to work best in most cases.  
Higher means a more global color match and wider bloom/bleed.  
Lower means a more local color match and smaller bloom/bleed. Too low and the reference clip will become visible.  
Test values 3 and 8 and this will become more clear.

__*`planes`* (optional)__  
Which planes to color fix. Any unmentioned planes will simply be copied.  
If not set, all planes will be color-fixed.

__*`backend`* (optional)__  
The used backend. **16-bit float input is always much faster on GPU, but not supported by older GPUs.**
* `cpu` CPU mode *(slow)*.
* `ncnn` GPU mode using NCNN. Works on almost any GPU, even Mac *(fast)*.
* `directml` GPU mode using DirectML. Works on most GPUs, Windows only *(fast)*.
* `tensorrt` GPU mode using TensorRT. Requires an Nvidia RTX GPU. On the first run, this mode will automatically build an engine, which may take a few minutes. Changing wavelets or input dimensions will trigger rebuilding, but build engines are stored *(very fast)*.

__*`num_streams`* (optional)__  
Number of parallel GPU streams. Higher can be faster, but requires more VRAM. Does not affect the CPU backend.

__*`gpu_id`* (optional)__  
Which GPU to use starting from 0. Can be used to switch between iGPU/dGPU. Does not affect the CPU backend.

__*`engine_folder`* (optional)__  
Optional path to the TensorRT engine storage location. By default engines are stored in `vs_colorfix/engines`. Only affects the TensorRT backend.

> [!TIP]
> If your clips are not sufficiently aligned or synchronized, use [vs_align](https://github.com/pifroggi/vs_align) to align them first.

<br />

## Guided Color Fix
Fixes color shift guided by a trained AI model that intelligently transfer colors from a reference while avoiding the bleed/bloom produced by the Average and Wavelet Color Fix when the shift is not uniform, but is much slower.

```python
import vs_colorfix
clip = vs_colorfix.guided(clip, ref, planes=[0, 1, 2], backend="tensorrt", num_streams=1, gpu_id=0, engine_folder=None)
```

__*`clip`*__  
Base clip where the colors will be applied to.  
Must be in float format.

__*`ref`*__  
Reference clip where the colors are taken from.  
Check the [comparisons](https://slow.pics/c/abXjnKn3) to get an idea how close it should match the base clip.

__*`planes`* (optional)__  
Which planes to color fix. Any unmentioned planes will simply be copied.  
If not set, all planes will be color-fixed.

__*`backend`* (optional)__  
The used backend.
* `cpu` CPU mode *(very slow)*.
* `ncnn` GPU mode using NCNN. Works on almost any GPU, even Mac *(fast)*.
* `directml` GPU mode using DirectML. Works on most GPUs, Windows only *(faster)*.
* `tensorrt` GPU mode using TensorRT. Requires an Nvidia RTX GPU. On the first run, this mode will automatically build an engine, which may take a few minutes. Changing input dimensions will trigger rebuilding, but build engines are stored *(very fast, low vram)*.

__*`num_streams`* (optional)__  
Number of parallel GPU streams. Higher can be faster, but requires more VRAM. Does not affect the CPU backend.

__*`gpu_id`* (optional)__  
Which GPU to use starting from 0. Can be used to switch between iGPU/dGPU. Does not affect the CPU backend.

__*`engine_folder`* (optional)__  
Optional path to the TensorRT engine storage location. By default engines are stored in `vs_colorfix/engines`. Only affects the TensorRT backend.

> [!TIP]
> If your clips are not sufficiently aligned or synchronized, use [vs_align](https://github.com/pifroggi/vs_align) to align them first.

<br />

## Benchmarks
Benchmarks were done on a RTX 4090 GPU and a Ryzen 5900X CPU with 16-bit input clips.

<table>
  <tr>
    <td align="center" valign="top">

<table>
  <thead>
    <tr align="center">
      <th colspan="5">Wavelet Color Fix</th>
    </tr>
    <tr align="center">
      <th>Resolution</th>
      <th>TensorRT</th>
      <th>DirectML</th>
      <th>NCNN</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>1440x1080</td>
      <td>~360 fps</td>
      <td>~250 fps</td>
      <td>~250 fps</td>
      <td>~20 fps</td>
    </tr>
    <tr align="center">
      <td>2880x2160</td>
      <td>~80 fps</td>
      <td>~60 fps</td>
      <td>~60 fps</td>
      <td>~5 fps</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr align="center">
      <th colspan="5">Guided Color Fix</th>
    </tr>
    <tr align="center">
      <th>Resolution</th>
      <th>TensorRT</th>
      <th>DirectML</th>
      <th>NCNN</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>1440x1080</td>
      <td>~52 fps</td>
      <td>~30 fps</td>
      <td>~20 fps</td>
      <td>~0.5 fps</td>
    </tr>
    <tr align="center">
      <td>2880x2160</td>
      <td>~13 fps</td>
      <td>~8 fps</td>
      <td>~5 fps</td>
      <td>~0.2 fps</td>
    </tr>
  </tbody>
</table>

</td>

<td align="center" valign="top">

<table>
  <thead>
    <tr align="center">
      <th colspan="3">Average Color Fix</th>
    </tr>
    <tr align="center">
      <th>Resolution</th>
      <th>fast=False</th>
      <th>fast=True</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>1440x1080</td>
      <td>~250 fps</td>
      <td>~850 fps</td>
    </tr>
    <tr align="center">
      <td>2880x2160</td>
      <td>~60 fps</td>
      <td>~150 fps</td>
    </tr>
  </tbody>
</table>

</td>
  </tr>
</table>

<br />

## Acknowledgements
Average Color Fix idea from [chaiNNer](https://github.com/chaiNNer-org/chaiNNer).  
Wavelet Color Fix idea from [sd-webui-stablesr](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py).  
Guided Color Fix architecture created and model training by [Bendel](https://huggingface.co/labx).
