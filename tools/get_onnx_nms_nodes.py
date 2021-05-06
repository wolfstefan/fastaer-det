import sys
import onnx

model_path = sys.argv[1]
model = onnx.load_model(model_path)
for n in model.graph.node:
    if 'NonMax' in n.op_type:
        print(n.output)
        break
