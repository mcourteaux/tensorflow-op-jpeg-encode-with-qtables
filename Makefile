CC=g++-12

all: compress_op.so

TF_INSTALL_DIR=$(shell pip3 show tensorflow | grep Location | cut -d\  -f 2)/tensorflow
TF_LINK_FLAGS=-L$(TF_INSTALL_DIR) -l:libtensorflow_cc.so.2 -l:libtensorflow_framework.so.2
TF_COMPILE_FLAGS=-I$(TF_INSTALL_DIR)/include -D_GLIBCXX_USE_CXX11_ABI=1 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64

JPEG_FLAGS=$(shell pkg-config --cflags --libs libjpeg)

compress_op.so: compress_op.cc
	$(CC) -shared compress_op.cc $(TF_COMPILE_FLAGS) $(TF_LINK_FLAGS) $(JPEG_FLAGS) -fPIC -O2 -o $@

clean:
	rm -f compress_op.so test_input.png test_output_chromass.jpeg test_output_nochromass.jpg

test_input.png:
	wget https://seeklogo.com/images/U/ubuntu-linux-logo-A8280F4D05-seeklogo.com.png -O $@

test: compress_op.so test_input.png
	CUDA_VISIBLE_DEVICES="" python3 test.py
