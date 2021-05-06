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
#include "nmsPluginRotated.h"
#include "plugin_internal.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::DetectionOutputParameters;

namespace
{
const char* NMS_PLUGIN_VERSION{"1"};
const char* NMS_PLUGIN_NAME{"NMS_ROTATED_TRT_ONNX"};
} // namespace

PluginFieldCollection NMSRotatedPluginCreator::mFC{};
std::vector<PluginField> NMSRotatedPluginCreator::mPluginAttributes;

// Constrcutor
DetectionOutputRotated::DetectionOutputRotated(DetectionOutputParameters params)
    : param(params)
{
}

DetectionOutputRotated::DetectionOutputRotated(DetectionOutputParameters params, int C1, int C2, int numPriors)
    : param(params)
    , C1(C1)
    , C2(C2)
    , numPriors(numPriors)
{
}

// Parameterized constructor
DetectionOutputRotated::DetectionOutputRotated(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<DetectionOutputParameters>(d);
    // Channel size of the locData tensor
    // numPriors * numLocClasses * 5
    C1 = read<int>(d);
    // Channel size of the confData tensor
    // numPriors * param.numClasses
    C2 = read<int>(d);
    // Number of bounding boxes per sample
    numPriors = read<int>(d);
    ASSERT(d == a + length);
}

int DetectionOutputRotated::getNbOutputs() const
{
    // Plugin layer has 2 outputs
    return 2;
}

int DetectionOutputRotated::initialize()
{
    return STATUS_SUCCESS;
}

void DetectionOutputRotated::terminate() {}

// Returns output dimensions at given index
Dims DetectionOutputRotated::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 2);
    ASSERT(index == 0 || index == 1);
    // Output dimensions
    // index 0 : Dimensions 1x param.keepTopK x 11
    // index 1: Dimensions 1x1x1
    if (index == 0)
    {
        return DimsCHW(1, param.keepTopK, 11);
    }
    return DimsCHW(1, 1, 1);
}

// Returns the workspace size
size_t DetectionOutputRotated::getWorkspaceSize(int maxBatchSize) const
{
    return detectionInferenceRotatedWorkspaceSize(param.shareLocation, maxBatchSize, C1, C2, param.numClasses, numPriors,
        param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

// Plugin layer implementation
int DetectionOutputRotated::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // Input order {loc, conf, prior}
    const void* const locData = inputs[param.inputOrder[0]];
    const void* const confData = inputs[param.inputOrder[1]];

    // Output from plugin index 0: topDetections index 1: keepCount
    void* topDetections = outputs[0];
    void* keepCount = outputs[1];

    pluginStatus_t status = detectionInferenceRotated(stream, batchSize, C1, C2, param.shareLocation,
        param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK,
        param.confidenceThreshold, param.nmsThreshold, DataType::kFLOAT, locData,
        DataType::kFLOAT, confData, keepCount, topDetections, workspace, param.isNormalized, param.confSigmoid);
    ASSERT(status == STATUS_SUCCESS);

    return 0;
}

// Returns the size of serialized parameters
size_t DetectionOutputRotated::getSerializationSize() const
{
    // DetectionOutputParameters, C1,C2,numPriors
    return sizeof(DetectionOutputParameters) + sizeof(int) * 3;
}

// Serialization of plugin parameters
void DetectionOutputRotated::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, C1);
    write(d, C2);
    write(d, numPriors);
    ASSERT(d == a + getSerializationSize());
}

// Check if the DataType and Plugin format is supported
bool DetectionOutputRotated::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kINT32) && format == PluginFormat::kNCHW);
}

// Get the plugin type
const char* DetectionOutputRotated::getPluginType() const
{
    return NMS_PLUGIN_NAME;
}

// Get the plugin version
const char* DetectionOutputRotated::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

// Clean up
void DetectionOutputRotated::destroy()
{
    delete this;
}

// Cloning the plugin
IPluginV2Ext* DetectionOutputRotated::clone() const
{
    // Create a new instance
    IPluginV2Ext* plugin = new DetectionOutputRotated(param, C1, C2, numPriors);

    // Set the namespace
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// Set plugin namespace
void DetectionOutputRotated::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* DetectionOutputRotated::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index.
DataType DetectionOutputRotated::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Two outputs
    ASSERT(index == 0 || index == 1);
    if (index == 0)
        return DataType::kFLOAT;
    return DataType::kINT32;
}

// Return true if output tensor is broadcast across a batch.
bool DetectionOutputRotated::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DetectionOutputRotated::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return true;
}

// Configure the layer with input and output data types.
// inutDims: input Dimensions for the plugin layer
// nInputs : Number of inputs to the plugin layer
// outputDims: output Dimensions from the plugin layer
// nOutputs: number of outputs from the plugin layer
// type: DataType configuration for the plugin layer
// format: format NCHW, NHWC etc
// maxbatchSize: maximum batch size for the plugin layer
void DetectionOutputRotated::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    // Number of input dimension should be 3
    ASSERT(nbInputs == 2);

    // Number of output dimension wil be 2
    ASSERT(nbOutputs == 2);

    // Verify all the input dimensions
    for (int i = 0; i < nbInputs; i++)
    {
        ASSERT(inputDims[i].nbDims == 3);
    }

    // Verify all the output dimensions
    for (int i = 0; i < nbOutputs; i++)
    {
        ASSERT(outputDims[i].nbDims == 3);
    }

    // Configure C1, C2 and numPriors
    // Input ordering  C1, C2, numPriors
    C1 = inputDims[param.inputOrder[0]].d[0];
    C2 = inputDims[param.inputOrder[1]].d[0];

    const int nbBoxCoordinates = 5;
    ASSERT(param.shareLocation);
    numPriors = inputDims[param.inputOrder[0]].d[0] / nbBoxCoordinates;
    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;

    auto d = inputDims[param.inputOrder[0]].d;

    // Verify C1
    ASSERT(numPriors * numLocClasses * nbBoxCoordinates == inputDims[param.inputOrder[0]].d[0]);

    // Verify C2
    ASSERT(numPriors * param.numClasses == inputDims[param.inputOrder[1]].d[0]);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DetectionOutputRotated::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void DetectionOutputRotated::detachFromContext() {}

// Plugin creator constructor
NMSRotatedPluginCreator::NMSRotatedPluginCreator()
{
    // NMS Plugin field meta data {name,  data, type, length}
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("confidenceThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("inputOrder", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("confSigmoid", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

// Returns the plugin name
const char* NMSRotatedPluginCreator::getPluginName() const
{
    return NMS_PLUGIN_NAME;
}

// Returns the plugin version
const char* NMSRotatedPluginCreator::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

// Returns the plugin field names
const PluginFieldCollection* NMSRotatedPluginCreator::getFieldNames()
{
    return &mFC;
}

// Creates the NMS plugin
IPluginV2Ext* NMSRotatedPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    // Default init values for TF SSD network
    params.inputOrder[0] = 0;
    params.inputOrder[1] = 1;
    params.inputOrder[2] = 2;

    // Read configurations from  each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "shareLocation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.shareLocation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.backgroundLabelId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.numClasses = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "topK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.topK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.keepTopK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confidenceThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.confidenceThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "nmsThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.nmsThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confSigmoid"))
        {
            params.confSigmoid = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "isNormalized"))
        {
            params.isNormalized = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "inputOrder"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            const int size = fields[i].length;
            const int* o = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.inputOrder[j] = *o;
                o++;
            }
        }
    }

    DetectionOutputRotated* obj = new DetectionOutputRotated(params);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* NMSRotatedPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call NMS::destroy()
    DetectionOutputRotated* obj = new DetectionOutputRotated(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
