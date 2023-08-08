# JPEG compression TensorFlowOp with custom quantization tables.

To build the op into `compress_op.so`, use:

```sh
make 
```

Then using the conventional way of loading custom ops in TensorFlow:

```py
import tensorflow as tf

module = tf.load_op_library("./compress_op.so")

img = ...
qtable_luma = ...
qtable_chroma = ...
encoded = module.encode_jpeg_with_qtables(img, qtable_luma, qtable_chroma, chroma_downsampling=True)
```

To test the implementation, there is a test script provided, which you can start with (which will download an additional test image from the internet):

```sh
make test
```

