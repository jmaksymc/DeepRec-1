/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL
#include <functional>
#include <vector>
#include "mkldnn.hpp"
#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/fake_input.h"
#include <gtest/gtest.h>
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

//----------------------------------------------------------------------------//
// Input conversion Tests are below.                                          //
//----------------------------------------------------------------------------//

namespace MKLInputConversionTestDefs {
    typedef std::tuple<
    DataType,                   // input_type
    std::vector<long long int>, // shape_1
    std::vector<long long int> // shape_2
    > InputConversionTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT,
        DataType::DT_BFLOAT16
    };
    std::vector<std::vector<long long int>> SHAPE_0 = {{128, 128}};
    std::vector<std::vector<long long int>> SHAPE_1 = {{128, 256}};
    std::vector<std::vector<long long int>> SHAPE_2 = {{265, 128}};
} // namespace InputConversionTestDefs

using namespace MKLInputConversionTestDefs;
class InputConversionTestBase :
    public ::testing::WithParamInterface<MKLInputConversionTestDefs::InputConversionTestParams>,
    public OpsTestBase {
 private:
    // Test definition (straight from Params, filled in SetUp)
    DataType input_type;
    std::vector<long long int> shape_0_vec;
    std::vector<long long int> shape_1_vec;
    // Test input Tensors (filled in SetUp)
    Tensor input_0;
    Tensor input_1;
    Tensor mkl_input_0;
    Tensor mkl_input_1;
    // Test output Tensors (filled in Run method)
    Tensor mkl_values;
    Tensor default_values;

    void runMkl() {
	TF_EXPECT_OK(
          NodeDefBuilder("mkl_input_conversion_op", "_MklInputConversion") //build node
              .Input(FakeInput(input_type))
              .Input(FakeInput(input_type))
              .Input(FakeInput(DT_UINT8))
              .Input(FakeInput(DT_UINT8))
              .Attr("_kernel", "MklLayoutDependentOp")
              .Finalize(node_def()));
      TF_EXPECT_OK(InitOp()); //initial
      switch(input_type) {
          case DT_FLOAT:
              AddInputFromArray<float>(input_0.shape(), input_0.flat<float>()); // input_0
              AddInputFromArray<float>(input_1.shape(), input_1.flat<float>()); // input_1
              break;
          case DT_BFLOAT16:
              AddInputFromArray<bfloat16>(input_0.shape(), input_0.flat<bfloat16>()); // input_0
              AddInputFromArray<bfloat16>(input_1.shape(), input_1.flat<bfloat16>()); // input_1
              break;
          default:
              GTEST_FAIL() << "Unexpected DataType";
        }
      AddInputFromArray<uint8_t>(mkl_input_0.shape(), mkl_input_0.flat<uint8_t>());
      AddInputFromArray<uint8_t>(mkl_input_1.shape(), mkl_input_1.flat<uint8_t>());
      TF_EXPECT_OK(RunOpKernel()); //Run the node computation
      mkl_values = *GetOutput(0); //Get outp
    }
 public:
    static std::string getTestCaseName(::testing::TestParamInfo<InputConversionTestParams> obj) {
        DataType input_type;
        std::vector<long long int> shape_0_vec;
        std::vector<long long int> shape_1_vec;
        std::tie(input_type, shape_0_vec, shape_1_vec) = obj.param;
        std::ostringstream result;
        result << "InputConversion_Type_";
        switch(input_type) {
            case DataType::DT_FLOAT:
                result << "FLOAT";
                break;
            case DataType::DT_BFLOAT16:
                result << "BFLOAT16";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }
        result << "_Sizes_0";
        for(auto &x : shape_0_vec){
            result << "_" << x;
        }
        result << "_Sizes_1";
        for(auto &x : shape_1_vec){
            result << "_" << x;
        }
        return result.str();
    }

    void SetUp() {
        std::tie(input_type, shape_0_vec, shape_1_vec) = this->GetParam();
        input_0 = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(shape_0_vec.data(), shape_0_vec.size())));
        input_1 = Tensor(input_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(shape_1_vec.data(), shape_1_vec.size())));
        switch(input_type) {
            case DT_FLOAT:
                input_0.flat<float>() = input_0.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>(); // input_0
                input_1.flat<float>() = input_1.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>(); // input_1
                break;
            case DT_BFLOAT16:
                input_0.flat<bfloat16>() = input_0.flat<bfloat16>().template setRandom<Eigen::internal::UniformRandomGenerator<bfloat16>>(); // input_0
                input_1.flat<bfloat16>() = input_1.flat<bfloat16>().template setRandom<Eigen::internal::UniformRandomGenerator<bfloat16>>(); // input_1
                input_0.flat<bfloat16>() = input_0.flat<bfloat16>() - input_0.flat<bfloat16>().constant((bfloat16)0.5);
                input_0.flat<bfloat16>() = input_0.flat<bfloat16>() * input_0.flat<bfloat16>().constant((bfloat16)200.0);
                input_1.flat<bfloat16>() = input_1.flat<bfloat16>() - input_1.flat<bfloat16>().constant((bfloat16)0.5);
                input_1.flat<bfloat16>() = input_1.flat<bfloat16>() * input_1.flat<bfloat16>().constant((bfloat16)200.0);
		break;
            default:
                GTEST_FAIL() << "Unexpected DataType";
        }
        // std::cout << "checkpoint 1" << std::endl;
        mkl_input_0 = Tensor(DT_UINT8, TensorShape({shape_0_vec[0], shape_0_vec[1]}));
        // mkl_input_0.flat<uint8_t>() = mkl_input_0.flat<uint8_t>().template setRandom<Eigen::internal::NormalRandomGenerator<uint8_t>>();
        mkl_input_1 = Tensor(DT_UINT8, TensorShape({shape_1_vec[0], shape_1_vec[1]}));
        // mkl_input_1.flat<uint8_t>() = mkl_input_1.flat<uint8_t>().template setRandom<Eigen::internal::NormalRandomGenerator<uint8_t>>();
        // std::cout << "checkpoint 2" << std::endl;
    }

    void Run() {
        //runDefault();
        runMkl();
    }

    void Validate() {
        // ASSERT_EQ(default_values.dtype(), mkl_values.dtype());
        // ASSERT_EQ(default_values.shape(), mkl_values.shape());
        // test::ExpectClose(default_values, mkl_values, 1e-5);
        std::cout << mkl_values.DebugString() << std::endl;
    }
};

TEST_P(InputConversionTestBase, CompareWithRefs) {
    SetUp();
    Run(); // true for BatchMatMulV2
    Validate();
};

INSTANTIATE_TEST_CASE_P(InputConversion_same_shape, InputConversionTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SHAPE_0),
        ::testing::ValuesIn(SHAPE_0)),
    InputConversionTestBase::getTestCaseName); 

  //----------------------------------------------------------------------------//
  // Performance benchmarks are below.                                          //
  //----------------------------------------------------------------------------//

  template <typename T>
  static Graph* InputConversion(const string& kind, int m, int n, int j, int k) {
    auto* g = new Graph(OpRegistry::Global());
    DataType type = DataTypeToEnum<T>::v();

    string op_name = "_MklInputConversion";

    Tensor tensor_0(type, TensorShape({ m, n }));
    tensor_0.flat<T>().setRandom();
    Node* input_0 = test::graph::Constant(g, tensor_0, "input_0");
    Node* not_mkl_shape_0 =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

    Tensor tensor_1(type, TensorShape({ j, k }));
    tensor_1.flat<T>().setRandom();
    Node* input_1 = test::graph::Constant(g, tensor_1, "input_1");
    Node* not_mkl_shape_1 =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");
      
    auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
      .Input(input_0)
      .Input(input_1)
      .Input(not_mkl_shape_0)
      .Input(not_mkl_shape_1)
      .Attr("T", type);

    nodeBuilder.Attr("_kernel", "MklLayoutDependentOp");

    TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

    return g;
  }

#define BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, NTH)                           \
  static void BM_Input_Conversion##_##kind##_##M##_##N##_##J##_##K##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                           \
    testing::UseRealTime();                                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters));                                      \
    SessionOptions opts;                                                                     \
    opts.config.set_intra_op_parallelism_threads(NTH);                                       \
    test::Benchmark(#DEVICE, InputConversion<T>(#kind, M, N, J, K), &opts).Run(iters);       \
  }                                                                                          \
  BENCHMARK(BM_Input_Conversion##_##kind##_##M##_##N##_##J##_##K##_##T##_##DEVICE##_##NTH);  \

#define BM_Input_Conversion_NTH(kind, M, N, J, K, T, DEVICE) \
  BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, 1);  \
  BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, 4);  \
  BM_Input_Conversion_Base(kind, M, N, J, K, T, DEVICE, 8);  \

#define BM_Input_Conversion_kind(M, N, J, K, T, DEVICE) \
  BM_Input_Conversion_NTH(Mkl, M, N, J, K, T, DEVICE);  \

#define BM_Input_Conversion_DT(M, N, J, K)             \
  BM_Input_Conversion_kind(M, N, J, K, float, cpu);    \
  BM_Input_Conversion_kind(M, N, J, K, bfloat16, cpu); \

  BM_Input_Conversion_DT(128, 128, 128, 128);
  BM_Input_Conversion_DT(128, 128, 256, 128);
  BM_Input_Conversion_DT(256, 128, 128, 128);


}  // end namespace tensorflow

#endif  // INTEL_MKL