import tensorflow as tf

print("Load custom JPEG compression library")
module = tf.load_op_library("./jpeg_encode_with_qtables.so")
print()


print("Load test image")
test_file = tf.io.read_file("test_input.png")
img = tf.io.decode_png(test_file, 3)
print()

qtable_luma = tf.random.uniform(shape=(8, 8), minval=1, maxval=255, dtype=tf.dtypes.int32)
qtable_luma = tf.cast(qtable_luma, tf.dtypes.uint32)
print("QTable Luma:")
print(qtable_luma.numpy())

qtable_chroma = tf.random.uniform(shape=(8, 8), minval=1, maxval=255, dtype=tf.dtypes.int32)
qtable_chroma = tf.cast(qtable_chroma, tf.dtypes.uint32)
print("QTable Chroma:")
print(qtable_chroma.numpy())

encoded = module.encode_jpeg_with_qtables(img, qtable_luma, qtable_chroma, chroma_downsampling=True)
print("Encoded size (chromass=True):", tf.strings.length(encoded))
tf.io.write_file("test_output_chromass.jpg", encoded)

encoded = module.encode_jpeg_with_qtables(img, qtable_luma, qtable_chroma, chroma_downsampling=False)
print("Encoded size (chromass=False):", tf.strings.length(encoded))
tf.io.write_file("test_output_nochromass.jpg", encoded)
