#include <setjmp.h>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/jpeg/jpeg_handle.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"

extern "C" {
#include "jerror.h"   // from @libjpeg_turbo   // IWYU pragma: export
#include "jpeglib.h"  // from @libjpeg_turbo  // IWYU pragma: export
}

using namespace tensorflow;

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status EncodeImageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused));
  c->set_output(0, c->Scalar());
  return OkStatus();
}

}  // namespace
}  // namespace tensorflow

REGISTER_OP("EncodeJpegWithQtables")
    .Input("images: uint8")
    .Input("qtable_luma: uint32")
    .Input("qtable_chroma: uint32")
    .Input("chroma_downsampling: bool")
    .Output("contents: string")
    .SetShapeFn(EncodeImageShapeFn);

namespace compress_with_qtables {

bool check_qtable_shape(const Tensor& qtable) {
  return qtable.dims() == 2 && qtable.dim_size(0) == 8 &&
         qtable.dim_size(1) == 8;
}

bool CompressInternal(const uint8* srcdata, int width, int height,
                      int components, bool optimize_jpeg_size, bool progressive,
                      bool chroma_downsampling, const unsigned int* qtable_luma,
                      const unsigned int* qtable_chroma, tstring* output) {
  if (output == nullptr) {
    LOG(ERROR) << "Output buffer is null: ";
    return false;
  }

  output->clear();

  int64_t total_size =
      static_cast<int64_t>(width) * static_cast<int64_t>(height);
  // Some of the internal routines do not gracefully handle ridiculously
  // large images, so fail fast.
  if (width <= 0 || height <= 0) {
    LOG(ERROR) << "Invalid image size: " << width << " x " << height;
    return false;
  }
  if (total_size >= (1LL << 29)) {
    LOG(ERROR) << "Image too large: " << total_size;
    return false;
  }

  int in_stride = width * components;

  JOCTET* buffer = nullptr;

  // NOTE: for broader use xmp_metadata should be made a Unicode string
  CHECK(srcdata != nullptr);
  CHECK(output != nullptr);
  // This struct contains the JPEG compression parameters and pointers to
  // working space
  struct jpeg_compress_struct cinfo;
  // This struct represents a JPEG error handler.
  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;  // recovery point in case of error

  // Step 1: allocate and initialize JPEG compression object
  // Use the usual jpeg error manager.
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = tensorflow::jpeg::CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    output->clear();
    delete[] buffer;
    return false;
  }

  jpeg_create_compress(&cinfo);

  // Step 2: specify data destination
  // We allocate a buffer of reasonable size. If we have a small image, just
  // estimate the size of the output using the number of bytes of the input.
  // If this is getting too big, we will append to the string by chunks of 1MB.
  // This seems like a reasonable compromise between performance and memory.
  int bufsize = std::min(width * height * components, 1 << 20);
  buffer = new JOCTET[bufsize];
  tensorflow::jpeg::SetDest(&cinfo, buffer, bufsize, output);

  // Step 3: set parameters for compression
  cinfo.image_width = width;
  cinfo.image_height = height;
  switch (components) {
    case 1:
      cinfo.input_components = 1;
      cinfo.in_color_space = JCS_GRAYSCALE;
      break;
    case 3:
    case 4:
      cinfo.input_components = 3;
      cinfo.in_color_space = JCS_RGB;
      break;
    default:
      LOG(ERROR) << " Invalid components value " << components << std::endl;
      output->clear();
      delete[] buffer;
      return false;
  }
  jpeg_set_defaults(&cinfo);
  if (optimize_jpeg_size) {
    cinfo.optimize_coding = TRUE;
  }

  cinfo.density_unit = 1;  // JFIF code for pixel size units: 1 = in, 2 = cm
  cinfo.X_density = 300;   // Horizontal pixel density
  cinfo.Y_density = 300;   // Vertical pixel density

  // Set the LUMA table (tbl=0):
  jpeg_add_quant_table(&cinfo, 0, qtable_luma, 100, TRUE);
  // Set the LUMA table (tbl=0):
  jpeg_add_quant_table(&cinfo, 1, qtable_chroma, 100, TRUE);

  // jpeg_set_quality(&cinfo, flags.quality, TRUE);  // TODO replace!

  if (progressive) {
    jpeg_simple_progression(&cinfo);
  }

  if (!chroma_downsampling) {
    // Turn off chroma subsampling (it is on by default).  For more details on
    // chroma subsampling, see http://en.wikipedia.org/wiki/Chroma_subsampling.
    for (int i = 0; i < cinfo.num_components; ++i) {
      cinfo.comp_info[i].h_samp_factor = 1;
      cinfo.comp_info[i].v_samp_factor = 1;
    }
  }

  jpeg_start_compress(&cinfo, TRUE);

  // JSAMPLEs per row in image_buffer
  std::unique_ptr<JSAMPLE[]> row_temp(
      new JSAMPLE[width * cinfo.input_components]);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row_pointer[1];  // pointer to JSAMPLE row[s]
    const uint8* r = &srcdata[cinfo.next_scanline * in_stride];
    row_pointer[0] = reinterpret_cast<JSAMPLE*>(const_cast<JSAMPLE*>(r));
    CHECK_EQ(jpeg_write_scanlines(&cinfo, row_pointer, 1), 1u);
  }
  jpeg_finish_compress(&cinfo);

  // release JPEG compression object
  jpeg_destroy_compress(&cinfo);
  delete[] buffer;
  return true;
}

class EncodeJpegWithQTablesOp : public OpKernel {
 public:
  explicit EncodeJpegWithQTablesOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("image must be 3-dimensional",
                                        image.shape().DebugString()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(image.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument(
            "Cannot encode images with >= max int32 elements"));

    const int32_t dim_size0 = static_cast<int32>(image.dim_size(0));
    const int32_t dim_size1 = static_cast<int32>(image.dim_size(1));
    const int32_t dim_size2 = static_cast<int32>(image.dim_size(2));

    // Get Quantization Tables
    const Tensor& qtable_luma = context->input(1);
    OP_REQUIRES(
        context, check_qtable_shape(qtable_luma),
        errors::InvalidArgument("qtable_luma must be a matrix of size 8*8: ",
                                qtable_luma.shape().DebugString()));

    const Tensor& qtable_chroma = context->input(2);
    OP_REQUIRES(
        context, check_qtable_shape(qtable_chroma),
        errors::InvalidArgument("qtable_chroma must be a matrix of size 8*8: ",
                                qtable_luma.shape().DebugString()));

    const Tensor& chroma_downsampling_tensor = context->input(3);
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(chroma_downsampling_tensor.shape()),
                errors::InvalidArgument(
                    "chroma_downsampling must be a scalar bool Tensor: ",
                    chroma_downsampling_tensor.shape().DebugString()));
    bool chroma_downsampling = chroma_downsampling_tensor.scalar<bool>()();

    // Autodetect format.
    int channels;
    channels = dim_size2;
    if (channels == 1 || channels == 3) {
      // good!
    } else {
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument("image must have 1 or 3 channels, got ",
                                  image.shape().DebugString()));
    }

    // Encode image to jpeg string
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES(
        context,
        CompressInternal(
            image.flat<uint8>().data(), dim_size1, dim_size0, channels,
            /* optimize_coding */ false, /* progressive */ false,
            chroma_downsampling, qtable_luma.flat<uint32>().data(),
            qtable_chroma.flat<uint32>().data(), &output->scalar<tstring>()()),
        errors::Internal("JPEG encoding failed"));
  }
};

REGISTER_KERNEL_BUILDER(Name("EncodeJpegWithQtables").Device(DEVICE_CPU),
                        EncodeJpegWithQTablesOp);

}  // namespace compress_with_qtables
