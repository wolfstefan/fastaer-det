/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel.h"
#include "../bboxUtils.h"
#include "../plugin_internal.h"

__global__ void rotBboxToPoly(
        int const numBboxes,
        float const * rotBboxes,
        float * polys)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numBboxes;
         index += blockDim.x * gridDim.x)
    {
        float x = rotBboxes[index * 5 + 0];
        float y = rotBboxes[index * 5 + 1];
        float w = rotBboxes[index * 5 + 2] - 1;
        float h = rotBboxes[index * 5 + 3] - 1;
        float a = rotBboxes[index * 5 + 4];

        float w_h = w / 2.0f;
        float h_h = h / 2.0f;

        float cos = cosf(a);
        float sin = sinf(a);

        float x1 = x + cos * w_h - sin * -h_h;
        float x2 = x + cos * w_h - sin * h_h;
        float x3 = x + cos * -w_h - sin * h_h;
        float x4 = x + cos * -w_h - sin * -h_h;

        float y1 = y + sin * w_h + cos * -h_h;
        float y2 = y + sin * w_h + cos * h_h;
        float y3 = y + sin * -w_h + cos * h_h;
        float y4 = y + sin * -w_h + cos * -h_h;

        polys[index * 8 + 0] = x1;
        polys[index * 8 + 1] = y1;
        polys[index * 8 + 2] = x2;
        polys[index * 8 + 3] = y2;
        polys[index * 8 + 4] = x3;
        polys[index * 8 + 5] = y3;
        polys[index * 8 + 6] = x4;
        polys[index * 8 + 7] = y4;
    }
}

pluginStatus_t detectionInferenceRotated(
    cudaStream_t stream,
    const int N,
    const int C1,
    const int C2,
    const bool shareLocation,
    const int backgroundLabelId,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const float confidenceThreshold,
    const float nmsThreshold,
    const DataType DT_BBOX,
    const void* locData,
    const DataType DT_SCORE,
    const void* confData,
    void* keepCount,
    void* topDetections,
    void* workspace,
    bool isNormalized,
    bool confSigmoid)
{
    // Batch size * number bbox per sample * 5 = total number of bounding boxes * 5
    const int locCount = N * C1;
    /*
     * shareLocation
     * Bounding box are shared among all classes, i.e., a bounding box could be classified as any candidate class.
     * Otherwise
     * Bounding box are designed for specific classes, i.e., a bounding box could be classified as one certain class or not (binary classification).
     */
    const int numLocClasses = shareLocation ? 1 : numClasses;

    size_t bboxDataSize = detectionRotatedForwardBBoxDataSize(N, numPredsPerClass, DataType::kFLOAT);
    void* bboxDataRaw = workspace;

    const int BS = 512;
    const int GS = (numPredsPerClass + BS - 1) / BS;
    rotBboxToPoly<<<GS, BS, 0, stream>>>(numPredsPerClass, (float const *)locData, (float *)bboxDataRaw);
    CSC(cudaGetLastError(), STATUS_FAILURE);

    pluginStatus_t status;

    /*
     * bboxDataRaw format:
     * [batch size, numPriors (per sample), numLocClasses, 8]
     */
    // float for now
    void* bboxData;
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DataType::kFLOAT);
    void* bboxPermute = nextWorkspacePtr((int8_t*) bboxDataRaw, bboxDataSize);

    /*
     * After permutation, bboxData format:
     * [batch_size, numLocClasses, numPriors (per sample) (numPredsPerClass), 8]
     * This is equivalent to swapping axis
     */
    ASSERT(shareLocation);
    if (!shareLocation)
    {
        status = permuteData(stream,
                             locCount,
                             numLocClasses,
                             numPredsPerClass,
                             4,
                             DataType::kFLOAT,
                             false,
                             bboxDataRaw,
                             bboxPermute);
        ASSERT_FAILURE(status == STATUS_SUCCESS);
        bboxData = bboxPermute;
    }
    /*
     * If shareLocation, numLocClasses = 1
     * No need to permute data on linear memory
     */
    else
    {
        bboxData = bboxDataRaw;
    }
    /*
     * Conf data format
     * [batch size, numPriors * param.numClasses, 1, 1]
     */
    const int numScores = N * C2;
    size_t scoresSize = detectionForwardPreNMSSize(N, C2);
    void* scores = nextWorkspacePtr((int8_t*) bboxPermute, bboxPermuteSize);
    // need a conf_scores
    /*
     * After permutation, bboxData format:
     * [batch_size, numClasses, numPredsPerClass, 1]
     */
    status = permuteData(stream,
                         numScores,
                         numClasses,
                         numPredsPerClass,
                         1,
                         DataType::kFLOAT,
                         confSigmoid,
                         confData,
                         scores);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    size_t indicesSize = detectionForwardPreNMSSize(N, C2);
    void* indices = nextWorkspacePtr((int8_t*) scores, scoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*) indices, indicesSize);
    void* postNMSIndices = nextWorkspacePtr((int8_t*) postNMSScores, postNMSScoresSize);

    //size_t sortingWorkspaceSize = sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, FLOAT32);
    void* sortingWorkspace = nextWorkspacePtr((int8_t*) postNMSIndices, postNMSIndicesSize);
    // Sort the scores so that the following NMS could be applied.
    status = sortScoresPerClass(stream,
                                N,
                                numClasses,
                                numPredsPerClass,
                                backgroundLabelId,
                                confidenceThreshold,
                                DataType::kFLOAT,
                                scores,
                                indices,
                                sortingWorkspace);
    ASSERT_FAILURE(status == STATUS_SUCCESS);
    
    // NMS
    status = allClassNMSRotated(stream,
                         N,
                         numClasses,
                         numPredsPerClass,
                         topK,
                         nmsThreshold,
                         shareLocation,
                         isNormalized,
                         DataType::kFLOAT,
                         DataType::kFLOAT,
                         bboxData,
                         scores,
                         indices,
                         postNMSScores,
                         postNMSIndices,
                         false);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    // Sort the bounding boxes after NMS using scores
    status = sortScoresPerImage(stream,
                                N,
                                numClasses * topK,
                                DataType::kFLOAT,
                                postNMSScores,
                                postNMSIndices,
                                scores,
                                indices,
                                sortingWorkspace);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    // Gather data from the sorted bounding boxes after NMS
    status = gatherTopDetectionsRotated(stream,
                                 shareLocation,
                                 N,
                                 numPredsPerClass,
                                 numClasses,
                                 topK,
                                 keepTopK,
                                 DataType::kFLOAT,
                                 DataType::kFLOAT,
                                 indices,
                                 scores,
                                 bboxData,
                                 keepCount,
                                 topDetections);
    ASSERT_FAILURE(status == STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

