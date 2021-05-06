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
#include <vector>

template <typename T_BBOX>
__device__ T_BBOX bboxSizePolyHorizontal(
    const T_BBOX * bbox,
    const bool normalized)
{
    T_BBOX xmin = min(min(min(bbox[0], bbox[2]), bbox[4]), bbox[6]);
    T_BBOX xmax = max(max(max(bbox[0], bbox[2]), bbox[4]), bbox[6]);
    T_BBOX ymin = min(min(min(bbox[1], bbox[3]), bbox[5]), bbox[7]);
    T_BBOX ymax = max(max(max(bbox[1], bbox[3]), bbox[5]), bbox[7]);
    if (xmax < xmin || ymax < ymin)
    {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0;
    }
    else
    {
        T_BBOX width = xmax - xmin;
        T_BBOX height = ymax - ymin;
        if (normalized)
        {
            return width * height;
        }
        else
        {
            // If bbox is not within range [0, 1].
            return (width + 1) * (height + 1);
        }
    }
}

template <typename T_BBOX>
__device__ void intersectBboxHorizontal(
    T_BBOX * bbox1,
    T_BBOX * bbox2,
    Bbox<T_BBOX>* intersect_bbox)
{
    T_BBOX bbox1xmin = min(min(min(bbox1[0], bbox1[2]), bbox1[4]), bbox1[6]);
    T_BBOX bbox1xmax = max(max(max(bbox1[0], bbox1[2]), bbox1[4]), bbox1[6]);
    T_BBOX bbox1ymin = min(min(min(bbox1[1], bbox1[3]), bbox1[5]), bbox1[7]);
    T_BBOX bbox1ymax = max(max(max(bbox1[1], bbox1[3]), bbox1[5]), bbox1[7]);
    T_BBOX bbox2xmin = min(min(min(bbox2[0], bbox2[2]), bbox2[4]), bbox2[6]);
    T_BBOX bbox2xmax = max(max(max(bbox2[0], bbox2[2]), bbox2[4]), bbox2[6]);
    T_BBOX bbox2ymin = min(min(min(bbox2[1], bbox2[3]), bbox2[5]), bbox2[7]);
    T_BBOX bbox2ymax = max(max(max(bbox2[1], bbox2[3]), bbox2[5]), bbox2[7]);
    if (bbox2xmin > bbox1xmax || bbox2xmax < bbox1xmin || bbox2ymin > bbox1ymax || bbox2ymax < bbox1ymin)
    {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->xmin = T_BBOX(0);
        intersect_bbox->ymin = T_BBOX(0);
        intersect_bbox->xmax = T_BBOX(0);
        intersect_bbox->ymax = T_BBOX(0);
    }
    else
    {
        intersect_bbox->xmin = max(bbox1xmin, bbox2xmin);
        intersect_bbox->ymin = max(bbox1ymin, bbox2ymin);
        intersect_bbox->xmax = min(bbox1xmax, bbox2xmax);
        intersect_bbox->ymax = min(bbox1ymax, bbox2ymax);
    }
}

template <typename T_BBOX>
__device__ float jaccardOverlapHorizontal(
    T_BBOX * bbox1,
    T_BBOX * bbox2,
    const bool normalized)
{
    Bbox<T_BBOX> intersect_bbox;
    intersectBboxHorizontal(bbox1, bbox2, &intersect_bbox);
    float intersect_width, intersect_height;
    if (normalized)
    {
        intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
        intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
    }
    else
    {
        intersect_width = intersect_bbox.xmax - intersect_bbox.xmin + 1;
        intersect_height = intersect_bbox.ymax - intersect_bbox.ymin + 1;
    }
    if (intersect_width > 0 && intersect_height > 0)
    {
        float intersect_size = intersect_width * intersect_height;
        float bbox1_size = bboxSizePolyHorizontal(bbox1, normalized);
        float bbox2_size = bboxSizePolyHorizontal(bbox2, normalized);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    }
    else
    {
        return 0.;
    }
}

template <typename T_BBOX>
__device__ int sig(T_BBOX val)
{
    static T_BBOX const eps = 1E-8;
    return (val > eps) - (val < -eps);
}

template <typename T_BBOX>
__device__ T_BBOX cross(T_BBOX ox, T_BBOX oy, T_BBOX ax, T_BBOX ay, T_BBOX bx, T_BBOX by)
{
    return (ax - ox) * (by - oy) - (bx - ox) * (ay - oy);
}

template <typename T_BBOX>
__device__ T_BBOX bboxSizePoly(
        const T_BBOX * bbox,
        const bool normalized = false)
{
    T_BBOX result = 0;
    result += bbox[0] * bbox[3] - bbox[1] * bbox[2];
    result += bbox[2] * bbox[5] - bbox[3] * bbox[4];
    result += bbox[4] * bbox[7] - bbox[5] * bbox[6];
    result += bbox[6] * bbox[1] - bbox[7] * bbox[0];
    return result / 2;
}

template <typename T>
__device__ void swap(T & a, T & b)
{
    T temp = a;
    a = b;
    b = temp;
}

template <typename T_BBOX>
__device__ void reverseBbox(T_BBOX * bbox)
{
    swap(bbox[0], bbox[6]);
    swap(bbox[1], bbox[7]);
    swap(bbox[2], bbox[4]);
    swap(bbox[3], bbox[5]);
}

template <typename T_BBOX>
__device__ T_BBOX area(T_BBOX * ps, int n)
{
    ps[2 * n] = ps[0];
    ps[2 * n + 1] = ps[1];
    T_BBOX result = 0;
    for (int i = 0; i < n; i++)
    {
        result += ps[i * 2] * ps[(i + 1) * 2 + 1] - ps[i * 2 + 1] * ps[(i + 1) * 2];
    }
    return result / 2;
}

template <typename T_BBOX>
__device__ int lineCross(T_BBOX ax, T_BBOX ay, T_BBOX bx, T_BBOX by, T_BBOX cx, T_BBOX cy, T_BBOX dx, T_BBOX dy,
                         T_BBOX & px, T_BBOX & py)
{
    T_BBOX s1, s2;
    s1 = cross(ax, ay, bx, by, cx, cy);
    s2 = cross(ax, ay, bx, by, dx, dy);
    if (sig(s1) == 0 && sig(s2) == 0) return 2;
    if (sig(s2 - s1) == 0) return 0;
    px = (cx * s2 - dx * s1) / (s2 - s1);
    py = (cy * s2 - dy * s1) / (s2 - s1);
    return 1;
}

template <typename T_BBOX>
__device__ void polygon_cut(T_BBOX * p, int & n, T_BBOX ax, T_BBOX ay, T_BBOX bx, T_BBOX by, T_BBOX * pp)
{
    int m = 0;
    p[2 * n] = p[0];
    p[2 * n + 1] = p[1];
    for (int i = 0; i < n; i++)
    {
        if (sig(cross(ax, ay, bx, by, p[i * 2], p[i * 2 + 1])) > 0)
        {
            pp[2 * (m)] = p[i * 2];
            pp[2 * (m++) + 1] = p[i * 2 + 1];
        }
        if (sig(cross(ax, ay, bx, by, p[i * 2], p[i * 2 + 1])) !=
            sig(cross(ax, ay, bx, by, p[(i + 1) * 2], p[(i + 1) * 2 + 1])))
        {
            lineCross(ax, ay, bx, by, p[i * 2], p[i * 2 + 1], p[(i + 1) * 2], p[(i + 1) * 2 + 1], pp[2 * m], pp[2 * (m++) + 1]);
        }
    }
    n = 0;
    for (int i = 0; i < m; i++)
    {
        if (!i || !(sig(pp[2 * i] - pp[2 * (i - 1)]) == 0 && sig(pp[2 * i + 1] - pp[2 * (i - 1) + 1]) == 0))
        {
            p[2 * n] = pp[2 * i];
            p[2 * (n++) + 1] = pp[2 * i + 1];
        }
    }
    while (n > 1 && p[2 * (n - 1)] == p[0] && p[2 * (n - 1) + 1] == p[1]) n--;
}

template <typename T_BBOX>
__device__ T_BBOX intersectArea(T_BBOX ax, T_BBOX ay, T_BBOX bx, T_BBOX by,
                                T_BBOX cx, T_BBOX cy, T_BBOX dx, T_BBOX dy)
{
    auto zero = T_BBOX(0);
    int s1 = sig(cross(zero, zero, ax, ay, bx, by));
    int s2 = sig(cross(zero, zero, cx, cy, dx, dy));
    if (s1 == 0 || s2 == 0) return 0;
    if (s1 == -1)
    {
        swap(ax, bx);
        swap(ay, by);
    }
    if (s2 == -1)
    {
        swap(cx, dx);
        swap(cy, dy);
    }
    T_BBOX p[20] = {0, 0, ax, ay, bx, by};
    int n = 3;
    T_BBOX pp[102];
    polygon_cut(p, n, zero, zero, cx, cy, pp);
    polygon_cut(p, n, cx, cy, dx, dy, pp);
    polygon_cut(p, n, dx, dy, zero, zero, pp);
    T_BBOX result = fabs(area(p, n));
    if (s1 * s2 == -1) result = -result;
    return result;
}

template <typename T_BBOX>
__device__ T_BBOX intersectArea(
        T_BBOX * bbox1,
        T_BBOX * bbox2,
        int index)
{
    T_BBOX * bboxes[2] = {bbox1, bbox2};
    if (threadIdx.x % 16 <= 1)
    {
        auto id = threadIdx.x % 16;
        if (bboxSizePoly(bboxes[id]) < 0) reverseBbox(bboxes[id]);
    }
    unsigned warp_id = threadIdx.x % 32;
    unsigned mask = 0xFFFF;
    if (warp_id >= 16) mask <<= 16;
    __syncwarp(mask);

    float inter_area = intersectArea(bbox1[index / 4 * 2],           bbox1[index / 4 * 2 + 1],
                                     bbox1[(index / 4 + 1) % 4 * 2], bbox1[(index / 4 + 1) % 4 * 2 + 1],
                                     bbox2[index % 4 * 2],           bbox2[index % 4 * 2 + 1],
                                     bbox2[(index + 1) % 4 * 2],     bbox2[(index + 1) % 4 * 2 + 1]);
    for (int offset = 8; offset > 0; offset /= 2)
    {
      inter_area += __shfl_down_sync(mask, inter_area, offset);
    }
    return inter_area;
}

template <typename T_BBOX>
__device__ float jaccardOverlap(
        T_BBOX * bbox1,
        T_BBOX * bbox2,
        const bool normalized,
        int index)
{
    auto inter_area = intersectArea(bbox1, bbox2, index);
    T_BBOX iou = 0;

    if (threadIdx.x % 16 == 0)
    {
        T_BBOX union_area = fabs(bboxSizePoly(bbox1)) + fabs(bboxSizePoly(bbox2)) - inter_area;

        if (union_area == 0)
        {
            iou = (inter_area + 1) / (union_area + 1);
        }
        else
        {
            iou = inter_area / union_area;
        }
    }

    return iou;
}

template <typename T_BBOX>
__device__ void emptyBboxInfo(
    BboxInfo<T_BBOX>* bbox_info)
{
    bbox_info->conf_score = T_BBOX(0);
    bbox_info->label = -2; // -1 is used for all labels when shared_location is ture
    bbox_info->bbox_idx = -1;
    bbox_info->kept = false;
}
/********** new NMS for only score and index array **********/

template <typename T_SCORE, typename T_BBOX, int TSIZE>
__global__ void allClassNMSRotated_kernel(
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k,
    const float nms_threshold,
    const bool share_location,
    const bool isNormalized,
    T_BBOX* bbox_data, // bbox_data should be float to preserve location information
    T_SCORE* beforeNMS_scores,
    int* beforeNMS_index_array,
    T_SCORE* afterNMS_scores,
    int* afterNMS_index_array,
    bool flipXY = false)
{
    //__shared__ bool kept_bboxinfo_flag[CAFFE_CUDA_NUM_THREADS * TSIZE];
    extern __shared__ bool kept_bboxinfo_flag[];

    for (int i = 0; i < num; i++)
    {
        const int offset = i * num_classes * num_preds_per_class + blockIdx.x * num_preds_per_class;
        const int max_idx = offset + top_k; // put top_k bboxes into NMS calculation
        const int bbox_idx_offset = share_location ? (i * num_preds_per_class) : (i * num_classes * num_preds_per_class);

        // local thread data
        int loc_bboxIndex[TSIZE];
        T_BBOX loc_bbox[TSIZE * 8];

// initialize Bbox, Bboxinfo, kept_bboxinfo_flag
        // Eliminate shared memory RAW hazard  
        __syncthreads();
#pragma unroll
        for (int t = 0; t < TSIZE; t++)
        {
            const int cur_idx = (threadIdx.x + blockDim.x * t) / 16;
            const int item_idx = offset + cur_idx;

            if (item_idx < max_idx)
            {
                loc_bboxIndex[t] = beforeNMS_index_array[item_idx];

                if (loc_bboxIndex[t] != -1)
                {
                    const int bbox_data_idx = share_location ? (loc_bboxIndex[t] % num_preds_per_class + bbox_idx_offset) : loc_bboxIndex[t];

                    loc_bbox[t * 8]     = bbox_data[bbox_data_idx * 8 + 0];
                    loc_bbox[t * 8 + 1] = bbox_data[bbox_data_idx * 8 + 1];
                    loc_bbox[t * 8 + 2] = bbox_data[bbox_data_idx * 8 + 2];
                    loc_bbox[t * 8 + 3] = bbox_data[bbox_data_idx * 8 + 3];
                    loc_bbox[t * 8 + 4] = bbox_data[bbox_data_idx * 8 + 4];
                    loc_bbox[t * 8 + 5] = bbox_data[bbox_data_idx * 8 + 5];
                    loc_bbox[t * 8 + 6] = bbox_data[bbox_data_idx * 8 + 6];
                    loc_bbox[t * 8 + 7] = bbox_data[bbox_data_idx * 8 + 7];
                    if (threadIdx.x % 16 == 0)
                    {
                        kept_bboxinfo_flag[cur_idx] = true;
                    }
                }
                else
                {
                    if (threadIdx.x % 16 == 0)
                    {
                        kept_bboxinfo_flag[cur_idx] = false;
                    }
                }
            }
            else
            {
                if (threadIdx.x % 16 == 0)
                {
                    kept_bboxinfo_flag[cur_idx] = false;
                }
            }
        }

        __syncthreads();

        // filter out overlapped boxes with lower scores
        int ref_item_idx = offset;
        int ref_bbox_idx = share_location ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset) : beforeNMS_index_array[ref_item_idx];

        while ((ref_bbox_idx != -1) && ref_item_idx < max_idx)
        {
            // Eliminate shared memory RAW hazard
            __syncthreads();

            for (int t = 0; t < TSIZE; t++)
            {
                const int cur_idx = (threadIdx.x + blockDim.x * t) / 16;
                const int item_idx = offset + cur_idx;

                if ((kept_bboxinfo_flag[cur_idx]) && (item_idx > ref_item_idx))
                {
                    // TODO: may need to add bool normalized as argument, HERE true means normalized
                    if (jaccardOverlapHorizontal(&bbox_data[ref_bbox_idx * 8], &loc_bbox[t * 8], isNormalized) > nms_threshold &&
                        jaccardOverlap(&bbox_data[ref_bbox_idx * 8], &loc_bbox[t * 8], isNormalized, threadIdx.x % 16) > nms_threshold)
                    {
                        if (threadIdx.x % 16 == 0)
                        {
                            kept_bboxinfo_flag[cur_idx] = false;
                        }
                    }
                }
            }
            __syncthreads();

            do
            {
                ref_item_idx++;
            } while (ref_item_idx < max_idx && !kept_bboxinfo_flag[ref_item_idx - offset]);

            ref_bbox_idx = share_location ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset) : beforeNMS_index_array[ref_item_idx];
        }

        // store data
        for (int t = 0; t < TSIZE; t++)
        {
            if (threadIdx.x % 16 == 0)
            {
                const int cur_idx = (threadIdx.x + blockDim.x * t) / 16;
                const int read_item_idx = offset + cur_idx;
                const int write_item_idx = (i * num_classes * top_k + blockIdx.x * top_k) + cur_idx;
                /*
                 * If not not keeping the bbox
                 * Set the score to 0
                 * Set the bounding box index to -1
                 */
                if (read_item_idx < max_idx)
                {
                    afterNMS_scores[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? beforeNMS_scores[read_item_idx] : 0.0f;
                    afterNMS_index_array[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? loc_bboxIndex[t] : -1;
                }
            }
        }
    }
}

template <typename T_SCORE, typename T_BBOX>
pluginStatus_t allClassNMSRotated_gpu(
    cudaStream_t stream,
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k,
    const float nms_threshold,
    const bool share_location,
    const bool isNormalized,
    void* bbox_data,
    void* beforeNMS_scores,
    void* beforeNMS_index_array,
    void* afterNMS_scores,
    void* afterNMS_index_array,
    bool flipXY = false)
{
#define P(tsize) allClassNMSRotated_kernel<T_SCORE, T_BBOX, (tsize)>

    void (*kernel[8])(const int, const int, const int, const int, const float,
                      const bool, const bool, float*, T_SCORE*, int*, T_SCORE*,
                      int*, bool)
        = {
            P(16), P(32), P(48), P(64), P(80), P(96), P(112), P(128),
        };

    const int BS = 512;
    const int GS = num_classes;
    const int t_size = (top_k + BS - 1) / BS;

    kernel[t_size - 1]<<<GS, BS, BS * t_size * sizeof(bool), stream>>>(num, num_classes, num_preds_per_class,
                                                                       top_k, nms_threshold, share_location, isNormalized,
                                                                       (T_BBOX*) bbox_data,
                                                                       (T_SCORE*) beforeNMS_scores, 
                                                                       (int*) beforeNMS_index_array,
                                                                       (T_SCORE*) afterNMS_scores,
                                                                       (int*) afterNMS_index_array,
                                                                       flipXY);

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// allClassNMS LAUNCH CONFIG 
typedef pluginStatus_t (*nmsRotatedFunc)(cudaStream_t,
                               const int,
                               const int,
                               const int,
                               const int,
                               const float,
                               const bool,
                               const bool,
                               void*,
                               void*,
                               void*,
                               void*,
                               void*,
                               bool);

struct nmsRotatedLaunchConfigSSD
{
    DataType t_score;
    DataType t_bbox;
    nmsRotatedFunc function;

    nmsRotatedLaunchConfigSSD(DataType t_score, DataType t_bbox)
        : t_score(t_score)
        , t_bbox(t_bbox)
    {
    }
    nmsRotatedLaunchConfigSSD(DataType t_score, DataType t_bbox, nmsRotatedFunc function)
        : t_score(t_score)
        , t_bbox(t_bbox)
        , function(function)
    {
    }
    bool operator==(const nmsRotatedLaunchConfigSSD& other)
    {
        return t_score == other.t_score && t_bbox == other.t_bbox;
    }
};

static std::vector<nmsRotatedLaunchConfigSSD> nmsRotatedFuncVec;

bool nmsRotatedInit()
{
    nmsRotatedFuncVec.push_back(nmsRotatedLaunchConfigSSD(DataType::kFLOAT, DataType::kFLOAT,
                                            allClassNMSRotated_gpu<float, float>));
    return true;
}

static bool initialized = nmsRotatedInit();


pluginStatus_t allClassNMSRotated(cudaStream_t stream,
                        const int num,
                        const int num_classes,
                        const int num_preds_per_class,
                        const int top_k,
                        const float nms_threshold,
                        const bool share_location,
                        const bool isNormalized,
                        const DataType DT_SCORE,
                        const DataType DT_BBOX,
                        void* bbox_data,
                        void* beforeNMS_scores,
                        void* beforeNMS_index_array,
                        void* afterNMS_scores,
                        void* afterNMS_index_array,
                        bool flipXY)
{
    nmsRotatedLaunchConfigSSD lc = nmsRotatedLaunchConfigSSD(DT_SCORE, DT_BBOX, allClassNMSRotated_gpu<float, float>);
    for (unsigned i = 0; i < nmsRotatedFuncVec.size(); ++i)
    {
        if (lc == nmsRotatedFuncVec[i])
        {
            DEBUG_PRINTF("all class nms rotated kernel %d\n", i);
            return nmsRotatedFuncVec[i].function(stream,
                                          num,
                                          num_classes,
                                          num_preds_per_class,
                                          top_k,
                                          nms_threshold,
                                          share_location,
                                          isNormalized,
                                          bbox_data,
                                          beforeNMS_scores,
                                          beforeNMS_index_array,
                                          afterNMS_scores,
                                          afterNMS_index_array,
                                          flipXY);
        }
    }
    return STATUS_BAD_PARAM;
}
