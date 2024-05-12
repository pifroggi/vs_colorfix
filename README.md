# Color Fix functions for Vapoursynth

For example for transfering colors from one source to another, or fixing color shift from AI upscaling/restoration models. Also knows as Color Transfer sometimes. Example transfering DVD colors to upscaled image: https://imgsli.com/MjM5NzM5/0/2

### Requirements
* pip install numpy
* [pytorch](https://pytorch.org/) 

## Average Color Fix
Correct for color shift by matching the average color of a clip to that of a reference clip. This is a very fast way to transfer the colors from one clip to another that has spatially close to the same content, but different colors. Idea from [chaiNNer](https://github.com/chaiNNer-org/chaiNNer).

    import vs_colorfix
    clip = vs_colorfix.average(clip, reference_clip, blur_radius=5)

__*clip*__  
Clip where the color fix will be applied to.  
Must have a bit depth higher than 8 to avoid banding.

__*reference_clip*__  
Clip where the colors are taken from.  
Must have a bit depth higher than 8 to avoid banding.

__*blur_radius*__  
Blur will only be used internally and is not visible on the output.  
Higher means a more global color match. Wider bloom/bleed and less local color precision.  
Lower means a more local color match. Smaller bloom/bleed and more artifacts. Too low and the reference clip will become visible.

## Wavelet Color Fix
Correct for color shift by first separating a clip into wavelets of different frequencies, then matching the average color of that clip to that of a reference clip. This works similarly to the Average Color Fix, but produces better results at the cost of more computation. Both clips must have spatially close to the same content. The Wavelet Color Fix functions are from [sd-webui-stablesr](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py).  

    import vs_colorfix
    clip = vs_colorfix.wavelet(clip, reference_clip, wavelets=5)

__*clip*__  
Clip where the color fix will be applied to.  
Must be in RGBS or YUV444PS format.

__*reference_clip*__  
Clip where the colors are taken from.  
Must be in RGBS or YUV444PS format.

__*wavelets*__  
Number of wavelets, 5 seems to work best in most cases.  
Higher means a more global color match. Wider bloom/bleed and less local color precision.  
Lower means a more local color match. Smaller bloom/bleed and more artifacts. Too low and the reference clip will become visible.

## Tips
If your clips are not perfectly aligned, try this: https://github.com/pifroggi/vs_align
