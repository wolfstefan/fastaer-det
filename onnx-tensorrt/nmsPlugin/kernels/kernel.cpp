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
#include "../plugin.h"
#include "../plugin_internal.h"

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
    int topK, DataType DT_BBOX, DataType DT_SCORE)
{
    size_t wss[8];
    wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
    wss[2] = detectionForwardPreNMSSize(N, C2);
    wss[3] = detectionForwardPreNMSSize(N, C2);
    wss[4] = N * topK * 7 * 4;
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[7] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE),
        sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 8);
}

size_t detectionInferenceRotatedWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
                                              int topK, DataType DT_BBOX, DataType DT_SCORE)
{
    size_t wss[8];
    wss[0] = detectionRotatedForwardBBoxDataSize(N, numPredsPerClass, DT_BBOX);
    ASSERT(shareLocation);
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
    wss[2] = detectionForwardPreNMSSize(N, C2);
    wss[3] = detectionForwardPreNMSSize(N, C2);
    wss[4] = N * topK * 11 * 4;
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[7] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE),
                      sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 8);
}
