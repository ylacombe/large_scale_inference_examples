# large_scale_inference_examples
Large scale audio inference examples using `accelerate` and `datasets`.

Running large scale inference with audio datasets can be tricky:
- It's very costly/time-consuming to read and write audio datasets. Saving audio files might actually be the time bottleneck in most of the examples here.
- The higher the sampling rate, the more difficult it is to pre-load batches of audio during inference.

 
