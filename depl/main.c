#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "onnxruntime_c_api.h"

// Simple error handling
#define ORT_ABORT_ON_ERROR(expr)                                     \
  do {                                                               \
    OrtStatus* onnx_status = (expr);                                 \
    if (onnx_status != NULL) {                                       \
      const char* msg = g_ort->GetErrorMessage(onnx_status);         \
      fprintf(stderr, "ONNXRuntime Error: %s\n", msg);               \
      g_ort->ReleaseStatus(onnx_status);                             \
      exit(1);                                                       \
    }                                                                \
  } while (0)

static const OrtApi* g_ort = NULL;

static double now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.onnx> [iters]\n", argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  int iters = (argc >= 3) ? atoi(argv[2]) : 1000;

  // Initialize ONNX Runtime
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ppo_bench", &env));

  // Session options
  OrtSessionOptions* sess_opt = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&sess_opt));
  // Single-threaded for determinism in RT context
  ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(sess_opt, 1));
  ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(sess_opt, ORT_ENABLE_ALL));

  // Create session
  OrtSession* session = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, sess_opt, &session));
  g_ort->ReleaseSessionOptions(sess_opt);

  // Inspect I/O
  size_t num_inputs = 0, num_outputs = 0;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &num_inputs));
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &num_outputs));
  if (num_inputs != 1 || num_outputs != 1) {
    fprintf(stderr, "Expected 1 input and 1 output, got %zu and %zu\n", num_inputs, num_outputs);
    return 1;
  }

  OrtAllocator* allocator = NULL;
  ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  // Get I/O names
  char* input_name = NULL;
  {
    OrtAllocatedStringPtr name = g_ort->GetSessionInputNameAllocated(session, 0, allocator);
    input_name = (char*)name.get();
  }
  char* output_name = NULL;
  {
    OrtAllocatedStringPtr name = g_ort->GetSessionOutputNameAllocated(session, 0, allocator);
    output_name = (char*)name.get();
  }
  printf("Model loaded. Input: %s | Output: %s\n", input_name, output_name);

  // Assume obs_dim=16, act_dim=4 (adjust if different)
  const int obs_dim = 16;
  const int act_dim = 4;

  // Create input tensor (1 x obs_dim)
  int64_t input_shape[2] = {1, obs_dim};
  size_t input_tensor_size = (size_t)obs_dim;
  float* input_data = (float*)malloc(sizeof(float) * input_tensor_size);
  if (!input_data) { fprintf(stderr, "alloc failed\n"); return 1; }

  // Pseudo observation (deterministic)
  for (int i = 0; i < obs_dim; ++i) {
    input_data[i] = (float)((i % 4 == 0) ? 0.5 : (i % 4 == 1) ? 0.2 : (i % 4 == 2) ? 0.7 : 0.01);
  }

  OrtMemoryInfo* mem_info = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
  OrtValue* input_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
      mem_info, input_data, sizeof(float) * input_tensor_size, input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

  int is_tensor = 0;
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
  if (!is_tensor) { fprintf(stderr, "Input is not a tensor\n"); return 1; }

  // Prepare output
  OrtValue* output_tensor = NULL;
  const char* input_names[] = { input_name };
  const char* output_names[] = { output_name };

  // Warmup
  for (int i = 0; i < 10; ++i) {
    ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL,
                                  input_names, (const OrtValue* const*)&input_tensor, 1,
                                  output_names, 1, &output_tensor));
    g_ort->ReleaseValue(output_tensor);
    output_tensor = NULL;
  }

  // Benchmark loop
  double t0 = now_ns();
  for (int i = 0; i < iters; ++i) {
    ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL,
                                  input_names, (const OrtValue* const*)&input_tensor, 1,
                                  output_names, 1, &output_tensor));
    g_ort->ReleaseValue(output_tensor);
    output_tensor = NULL;
  }
  double t1 = now_ns();
  double avg_us = (t1 - t0) / (double)iters / 1e3;
  printf("Average inference over %d runs: %.3f us\n", iters, avg_us);

  // One final run to fetch and print outputs
  ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL,
                                input_names, (const OrtValue* const*)&input_tensor, 1,
                                output_names, 1, &output_tensor));
  float* out_data = NULL;
  ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&out_data));

  // Inspect output shape
  OrtTensorTypeAndShapeInfo* out_info = NULL;
  ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &out_info));
  size_t out_dim_count = 0;
  ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(out_info, &out_dim_count));
  int64_t out_shape[4] = {0};
  ORT_ABORT_ON_ERROR(g_ort->GetDimensions(out_info, out_shape, out_dim_count));
  printf("Output shape: [");
  for (size_t i = 0; i < out_dim_count; ++i) printf("%lld%s", (long long)out_shape[i], (i+1<out_dim_count)?", ":"");
  printf("]\n");
  g_ort->ReleaseTensorTypeAndShapeInfo(out_info);

  // Print first action vector
  int out_elems = 1;
  for (size_t i = 0; i < out_dim_count; ++i) out_elems *= (int)out_shape[i];
  int print_elems = out_elems < act_dim ? out_elems : act_dim;
  printf("Action (first %d elems):", print_elems);
  for (int i = 0; i < print_elems; ++i) {
    printf(" %.6f", out_data[i]);
  }
  printf("\n");

  // Cleanup
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseMemoryInfo(mem_info);
  g_ort->ReleaseValue(input_tensor);
  free(input_data);

  // Free names (allocated by ORT)
  allocator->Free(allocator, input_name);
  allocator->Free(allocator, output_name);

  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);

  return 0;
}
