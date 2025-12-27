import onnxruntime


ort_session = onnxruntime.InferenceSession(filepath)
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: np.random.randn(1, 64)}
ort_outs = ort_session.run(None, ort_inputs)