## STAN for video shadow detection

## Quick View

This is an official repo for STAN.  We use Diffusion for video shadow detection. 

Code will be released aspp. 

: )

## Note

* We conduct all the experiments with 4GPUs to get stable results, but when we transfer to single or two GPUs the performance gets decrease. This is may caused by learning rates when deploy **[accelerate](https://huggingface.co/docs/accelerate/index)** or due to the **batch-size**.  We find larger batch size performs better,  24G may not enough. You may adjust it by custom ways. 