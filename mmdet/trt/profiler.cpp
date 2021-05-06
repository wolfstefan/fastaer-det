#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <pybind11/pybind11.h>

#include <NvInferRuntime.h>

namespace py = pybind11;

class PYBIND11_EXPORT Profiler : public nvinfer1::IProfiler
{
	public:
		Profiler(std::function<void(std::string, float)> callback) :
			mCallback(callback)
		{}

		virtual void reportLayerTime(const char* layerName, float ms) override
		{
			mCallback(std::string(layerName), ms);
		}

	private:
		std::function<void(std::string, float)> mCallback;
};

PYBIND11_MODULE(profiler, m) {
	py::module::import("tensorrt");
	py::class_<Profiler, std::shared_ptr<Profiler>, nvinfer1::IProfiler>(m, "Profiler")
		.def(py::init<std::function<void(std::string, float)>>());
}
