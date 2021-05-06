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
#include <vector>
#include "../plugin_internal.h"
#include "kernel.h"

template <typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void gatherTopDetectionsRotated_kernel(
        const bool shareLocation,
        const int numImages,
        const int numPredsPerClass,
        const int numClasses,
        const int topK,
        const int keepTopK,
        const int* indices,
        const T_SCORE* scores,
        const T_BBOX* bboxData,
        int* keepCount,
        T_BBOX* topDetections)
{
    if (keepTopK > topK)
        return;
    for (int i = blockIdx.x * nthds_per_cta + threadIdx.x;
         i < numImages * keepTopK;
         i += gridDim.x * nthds_per_cta)
    {
        const int imgId = i / keepTopK;
        const int detId = i % keepTopK;
        const int offset = imgId * numClasses * topK;
        const int index = indices[offset + detId];
        const T_SCORE score = scores[offset + detId];
        /*
         * It is also likely that there is "bad bounding boxes" in the keepTopK bounding boxes.
         * We set the bounding boxes parameters as the parameters shown below.
         * These data will only show up at the end of keepTopK bounding boxes since the bounding boxes were sorted previously.
         * It is also not going to affect the count of valid bounding boxes (keepCount).
         * These data will probably never be used (because we have keepCount).
         */
        if (index == -1)
        {
            topDetections[i * 11] = imgId;  // image id
            topDetections[i * 11 + 1] = -1; // label
            topDetections[i * 11 + 2] = 0;  // confidence score
            // score==0 will not pass the VisualizeBBox check
            topDetections[i * 11 + 3] = 0;   // bbox x1
            topDetections[i * 11 + 4] = 0;   // bbox y1
            topDetections[i * 11 + 5] = 0;   // bbox x2
            topDetections[i * 11 + 6] = 0;   // bbox y2
            topDetections[i * 11 + 7] = 0;   // bbox x3
            topDetections[i * 11 + 8] = 0;   // bbox y3
            topDetections[i * 11 + 9] = 0;   // bbox x4
            topDetections[i * 11 + 10] = 0;   // bbox y4
        }
        else
        {
            const int bboxOffset = imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
            const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                        : index % (numClasses * numPredsPerClass)) + bboxOffset) * 8;
            topDetections[i * 11] = imgId;                                                            // image id
            topDetections[i * 11 + 1] = (index % (numClasses * numPredsPerClass)) / numPredsPerClass; // label
            topDetections[i * 11 + 2] = score;                                                        // confidence score
            // bbox x1
            topDetections[i * 11 + 3] = bboxData[bboxId];
            // bbox y1
            topDetections[i * 11 + 4] = bboxData[bboxId + 1];
            // bbox x2
            topDetections[i * 11 + 5] = bboxData[bboxId + 2];
            // bbox y2
            topDetections[i * 11 + 6] = bboxData[bboxId + 3];
            // bbox x3
            topDetections[i * 11 + 7] = bboxData[bboxId + 4];
            // bbox y3
            topDetections[i * 11 + 8] = bboxData[bboxId + 5];
            // bbox x4
            topDetections[i * 11 + 9] = bboxData[bboxId + 6];
            // bbox y4
            topDetections[i * 11 + 10] = bboxData[bboxId + 7];
            // Atomic add to increase the count of valid keepTopK bounding boxes
            // Without having to do manual sync.
            atomicAdd(&keepCount[i / keepTopK], 1);
        }
    }
}

template <typename T_BBOX, typename T_SCORE>
pluginStatus_t gatherTopDetectionsRotated_gpu(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* keepCount,
    void* topDetections)
{
    cudaMemsetAsync(keepCount, 0, numImages * sizeof(int), stream);
    const int BS = 32;
    const int GS = 32;
    gatherTopDetectionsRotated_kernel<T_BBOX, T_SCORE, BS><<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass,
                                                                           numClasses, topK, keepTopK,
                                                                           (int*) indices, (T_SCORE*) scores, (T_BBOX*) bboxData,
                                                                           (int*) keepCount, (T_BBOX*) topDetections);

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// gatherTopDetectionsRotated LAUNCH CONFIG
typedef pluginStatus_t (*gtdRotatedFunc)(cudaStream_t,
                               const bool,
                               const int,
                               const int,
                               const int,
                               const int,
                               const int,
                               const void*,
                               const void*,
                               const void*,
                               void*,
                               void*);
struct gtdRotatedLaunchConfig
{
    DataType t_bbox;
    DataType t_score;
    gtdRotatedFunc function;

    gtdRotatedLaunchConfig(DataType t_bbox, DataType t_score)
        : t_bbox(t_bbox)
        , t_score(t_score)
    {
    }
    gtdRotatedLaunchConfig(DataType t_bbox, DataType t_score, gtdRotatedFunc function)
        : t_bbox(t_bbox)
        , t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const gtdRotatedLaunchConfig& other)
    {
        return t_bbox == other.t_bbox && t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::vector<gtdRotatedLaunchConfig> gtdRotatedFuncVec;

bool gtdRotatedInit()
{
    gtdRotatedFuncVec.push_back(gtdRotatedLaunchConfig(DataType::kFLOAT, DataType::kFLOAT,
                                         gatherTopDetectionsRotated_gpu<float, float>));
    return true;
}

static bool initialized = gtdRotatedInit();

pluginStatus_t gatherTopDetectionsRotated(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const DataType DT_BBOX,
    const DataType DT_SCORE,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* keepCount,
    void* topDetections)
{
    gtdRotatedLaunchConfig lc = gtdRotatedLaunchConfig(DT_BBOX, DT_SCORE);
    for (unsigned i = 0; i < gtdRotatedFuncVec.size(); ++i)
    {
        if (lc == gtdRotatedFuncVec[i])
        {
            DEBUG_PRINTF("gatherTopDetections rotated kernel %d\n", i);
            return gtdRotatedFuncVec[i].function(stream,
                                          shareLocation,
                                          numImages,
                                          numPredsPerClass,
                                          numClasses,
                                          topK,
                                          keepTopK,
                                          indices,
                                          scores,
                                          bboxData,
                                          keepCount,
                                          topDetections);
        }
    }
    return STATUS_BAD_PARAM;
}
