CXX ?= g++
PIP_PACKAGE ?= "tensorflow"

SO_NAME=jpeg_encode_with_qtables.so
all: $(SO_NAME)

TF_INSTALL_DIR=$(shell pip3 show $(PIP_PACKAGE) | grep Location | cut -d\  -f 2)/tensorflow
TF_LINK_FLAGS=-L$(TF_INSTALL_DIR) -l:libtensorflow_cc.so.2 -l:libtensorflow_framework.so.2
TF_COMPILE_FLAGS=-I$(TF_INSTALL_DIR)/include -D_GLIBCXX_USE_CXX11_ABI=1 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64

JPEG_FLAGS=$(shell pkg-config --cflags --libs libjpeg)

$(SO_NAME): jpeg_encode_with_qtables.cc
	$(CXX) -shared jpeg_encode_with_qtables.cc $(TF_COMPILE_FLAGS) $(TF_LINK_FLAGS) $(JPEG_FLAGS) -fPIC -O2 -o $@

clean:
	rm -f $(SO_NAME) test_input.png test_output_chromass.jpg test_output_nochromass.jpg

test_input.png:
	wget https://seeklogo.com/images/U/ubuntu-linux-logo-A8280F4D05-seeklogo.com.png -O $@

test: $(SO_NAME) test_input.png
	CUDA_VISIBLE_DEVICES="" python3 test.py
