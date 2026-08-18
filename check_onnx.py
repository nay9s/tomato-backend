import onnxruntime as ort

session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"]
)

print("INPUT")
for x in session.get_inputs():
    print(x.name, x.shape, x.type)

print("\nOUTPUT")
for x in session.get_outputs():
    print(x.name, x.shape, x.type)

print("\nONNX โหลดสำเร็จ")