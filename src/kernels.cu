#include <cfloat>
#include <cmath>
#include <cstddef>
#include <vector>

#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaximumHeadDimension = 256;

template <typename T>
__device__ __forceinline__ float toFloat(T value);

template <>
__device__ __forceinline__ float toFloat<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t hiddenDimension, float epsilon) {
  const size_t rowIndex = blockIdx.x;
  const size_t rowOffset = rowIndex * hiddenDimension;
  const int threadIndex = threadIdx.x;

  float squaredSum = 0.0f;
  for (size_t columnIndex = threadIndex; columnIndex < hiddenDimension;
       columnIndex += blockDim.x) {
    const float value = toFloat(input[rowOffset + columnIndex]);
    squaredSum += value * value;
  }

  __shared__ float reductionBuffer[kThreadsPerBlock];
  reductionBuffer[threadIndex] = squaredSum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIndex < stride) {
      reductionBuffer[threadIndex] += reductionBuffer[threadIndex + stride];
    }
    __syncthreads();
  }

  const float inverseRootMeanSquare =
      rsqrtf(reductionBuffer[0] / static_cast<float>(hiddenDimension) + epsilon);
  for (size_t columnIndex = threadIndex; columnIndex < hiddenDimension;
       columnIndex += blockDim.x) {
    const float normalizedValue = toFloat(input[rowOffset + columnIndex]) *
                                  inverseRootMeanSquare *
                                  toFloat(weight[columnIndex]);
    output[rowOffset + columnIndex] = fromFloat<T>(normalizedValue);
  }
}

template <typename T>
__global__ void flashAttentionKernel(
    const T* query, const T* key, const T* value, T* output,
    int batchSize, int targetSequenceLength, int sourceSequenceLength,
    int queryHeads, int keyValueHeads, int headDimension, bool isCausal) {
  const int attentionRowIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int attentionRowCount = batchSize * targetSequenceLength * queryHeads;
  if (attentionRowIndex >= attentionRowCount) {
    return;
  }

  const int rowsPerBatch = targetSequenceLength * queryHeads;
  const int batchIndex = attentionRowIndex / rowsPerBatch;
  const int rowWithinBatch = attentionRowIndex % rowsPerBatch;
  const int targetPosition = rowWithinBatch / queryHeads;
  const int queryHeadIndex = rowWithinBatch % queryHeads;
  const int keyValueHeadIndex =
      queryHeadIndex / (queryHeads / keyValueHeads);

  const int queryOffset =
      ((batchIndex * targetSequenceLength + targetPosition) * queryHeads +
       queryHeadIndex) *
      headDimension;
  float queryValues[kMaximumHeadDimension];
  float outputValues[kMaximumHeadDimension] = {};
  for (int dimensionIndex = 0; dimensionIndex < headDimension;
       ++dimensionIndex) {
    queryValues[dimensionIndex] =
        toFloat(query[queryOffset + dimensionIndex]);
  }

  const float attentionScale = 1.0f / sqrtf(static_cast<float>(headDimension));
  float maximumScore = -FLT_MAX;

  // Three passes keep softmax stable without materializing the score matrix.
  for (int keyPosition = 0; keyPosition < sourceSequenceLength; ++keyPosition) {
    if (isCausal && keyPosition > targetPosition) {
      continue;
    }

    const int keyValueOffset =
        ((batchIndex * sourceSequenceLength + keyPosition) * keyValueHeads +
         keyValueHeadIndex) *
        headDimension;
    float dotProduct = 0.0f;
    for (int dimensionIndex = 0; dimensionIndex < headDimension;
         ++dimensionIndex) {
      dotProduct = fmaf(queryValues[dimensionIndex],
                        toFloat(key[keyValueOffset + dimensionIndex]),
                        dotProduct);
    }
    maximumScore =
        fmaxf(maximumScore, attentionScale * dotProduct);
  }

  float softmaxDenominator = 0.0f;
  for (int keyPosition = 0; keyPosition < sourceSequenceLength; ++keyPosition) {
    if (isCausal && keyPosition > targetPosition) {
      continue;
    }

    const int keyOffset =
        ((batchIndex * sourceSequenceLength + keyPosition) * keyValueHeads +
         keyValueHeadIndex) *
        headDimension;
    float dotProduct = 0.0f;
    for (int dimensionIndex = 0; dimensionIndex < headDimension;
         ++dimensionIndex) {
      dotProduct = fmaf(queryValues[dimensionIndex],
                        toFloat(key[keyOffset + dimensionIndex]), dotProduct);
    }
    softmaxDenominator +=
        expf(attentionScale * dotProduct - maximumScore);
  }

  const float inverseSoftmaxDenominator =
      softmaxDenominator == 0.0f ? 0.0f : 1.0f / softmaxDenominator;
  for (int keyPosition = 0; keyPosition < sourceSequenceLength; ++keyPosition) {
    if (isCausal && keyPosition > targetPosition) {
      continue;
    }

    const int keyValueOffset =
        ((batchIndex * sourceSequenceLength + keyPosition) * keyValueHeads +
         keyValueHeadIndex) *
        headDimension;
    float dotProduct = 0.0f;
    for (int dimensionIndex = 0; dimensionIndex < headDimension;
         ++dimensionIndex) {
      dotProduct = fmaf(queryValues[dimensionIndex],
                        toFloat(key[keyValueOffset + dimensionIndex]),
                        dotProduct);
    }
    const float attentionWeight =
        inverseSoftmaxDenominator *
        expf(attentionScale * dotProduct - maximumScore);
    for (int dimensionIndex = 0; dimensionIndex < headDimension;
         ++dimensionIndex) {
      outputValues[dimensionIndex] =
          fmaf(attentionWeight,
               toFloat(value[keyValueOffset + dimensionIndex]),
               outputValues[dimensionIndex]);
    }
  }

  for (int dimensionIndex = 0; dimensionIndex < headDimension;
       ++dimensionIndex) {
    output[queryOffset + dimensionIndex] =
        fromFloat<T>(outputValues[dimensionIndex]);
  }
}

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  const size_t inputBytes = h_input.size() * sizeof(T);
  const size_t weightBytes = h_weight.size() * sizeof(T);
  const size_t outputBytes = h_output.size() * sizeof(T);

  T* deviceInput = nullptr;
  T* deviceWeight = nullptr;
  T* deviceOutput = nullptr;
  RUNTIME_CHECK(cudaMalloc(&deviceInput, inputBytes));
  RUNTIME_CHECK(cudaMalloc(&deviceWeight, weightBytes));
  RUNTIME_CHECK(cudaMalloc(&deviceOutput, outputBytes));
  RUNTIME_CHECK(cudaMemcpy(deviceInput, h_input.data(), inputBytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(deviceWeight, h_weight.data(), weightBytes,
                           cudaMemcpyHostToDevice));

  rmsNormKernel<T><<<static_cast<unsigned int>(rows), kThreadsPerBlock>>>(
      deviceInput, deviceWeight, deviceOutput, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), deviceOutput, outputBytes,
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(deviceInput));
  RUNTIME_CHECK(cudaFree(deviceWeight));
  RUNTIME_CHECK(cudaFree(deviceOutput));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  const size_t queryBytes = h_q.size() * sizeof(T);
  const size_t keyBytes = h_k.size() * sizeof(T);
  const size_t valueBytes = h_v.size() * sizeof(T);
  const size_t outputBytes = h_o.size() * sizeof(T);

  T* deviceQuery = nullptr;
  T* deviceKey = nullptr;
  T* deviceValue = nullptr;
  T* deviceOutput = nullptr;
  RUNTIME_CHECK(cudaMalloc(&deviceQuery, queryBytes));
  RUNTIME_CHECK(cudaMalloc(&deviceKey, keyBytes));
  RUNTIME_CHECK(cudaMalloc(&deviceValue, valueBytes));
  RUNTIME_CHECK(cudaMalloc(&deviceOutput, outputBytes));
  RUNTIME_CHECK(cudaMemcpy(deviceQuery, h_q.data(), queryBytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(deviceKey, h_k.data(), keyBytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(deviceValue, h_v.data(), valueBytes,
                           cudaMemcpyHostToDevice));

  const int attentionRowCount = batch_size * target_seq_len * query_heads;
  const int blockCount =
      (attentionRowCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
  flashAttentionKernel<T><<<blockCount, kThreadsPerBlock>>>(
      deviceQuery, deviceKey, deviceValue, deviceOutput, batch_size,
      target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), deviceOutput, outputBytes,
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(deviceQuery));
  RUNTIME_CHECK(cudaFree(deviceKey));
  RUNTIME_CHECK(cudaFree(deviceValue));
  RUNTIME_CHECK(cudaFree(deviceOutput));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
