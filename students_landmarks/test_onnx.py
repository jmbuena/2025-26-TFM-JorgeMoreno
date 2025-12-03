import cv2
import numpy as np
import onnxruntime
import torch

from src.dataloader import Mode, MyDataset
from images_framework.src.constants import Modes
from src.students_landmarks import StudentsLandmarks
from convert_to_onnx import load_first_annotation, parse_options


filepath = "ModelMerged.onnx"

ort_session = onnxruntime.InferenceSession(filepath)
input_name = ort_session.get_inputs()[0].name

unknown, args = parse_options()
anns_file = args.anns_file

sa = StudentsLandmarks('')
sa.parse_options(unknown)
sa.load(Modes.TEST)

anns = load_first_annotation(anns_file)
dataset_train = MyDataset(anns, sa.indices, sa.regressor, sa.width, sa.height, Mode.TEST)

img: np.ndarray = dataset_train[0]["img"]
img = img.astype(np.float32)
original_image = img.copy()

img = np.expand_dims(img, axis=0)
print(img.shape)

ort_outs = ort_session.run(None, {
	"x": img,
})

original_image = np.transpose(original_image, (1, 2, 0))
original_image = np.ascontiguousarray(original_image)
original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
print(original_image.shape, " -- ", original_image.dtype)

width, height, _ = original_image.shape

output = ort_outs[0][0]
for index in range(0, len(output), 2):
	x, y = int(output[index] * width), int(output[index + 1] * height)

	print((x, y))

	cv2.circle(original_image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

cv2.imshow("test", original_image)
cv2.waitKey(0)
