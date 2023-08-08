# JPEG compression TensorFlowOp with custom quantization tables.

## Building

To build the op into `jpeg_encode_with_qtables.so`, use:

```sh
make 
```

To test the implementation, there is a test script provided, which you can start with (which will download an additional test image from the internet):

```sh
make test
```

## Usage

Then using the conventional way of loading custom ops in TensorFlow:

```py
import tensorflow as tf

module = tf.load_op_library("./jpeg_encode_with_qtables.so")

img = ... # shape: (w, h, 3), uint8
qtable_luma = ... # shape: (8, 8), uint32
qtable_chroma = ... # shape: (8, 8), uint32
encoded = module.encode_jpeg_with_qtables(img, qtable_luma, qtable_chroma, chroma_downsampling=True)
```

**Note that the `qtable_luma` and `qtable_chroma` are of type `tf.uint32`, yet any sensible value to use is in the range `1` to `255`. Anything outside of this range will be clamped by the implementation to be JPEG baseline compliant.**

