import cv2
import numpy as np
import onnxruntime

from images_framework.src.constants import Modes
from src.students_landmarks import StudentsLandmarks
from src.utils import get_one_image_from_dataset, parse_common_onnx_options


def main():
	unknown, args = parse_common_onnx_options()
	anns_file = args.anns_file

	# Load the model
	sa = StudentsLandmarks('')
	sa.parse_options(unknown)
	sa.load(Modes.TEST)

	# Get one image to test the dataset
	img = get_one_image_from_dataset(anns_file=anns_file, studentLandmarks=sa)
	original_image = img.copy()

	# Add the batch dimension to the image
	img = np.expand_dims(img, axis=0)

	# Start & run the inference
	ort_session = onnxruntime.InferenceSession(args.onnx_path)
	ort_outs = ort_session.run(None, {
		"x": img,
	})
	output = ort_outs[0][0]

	# Fix the original image to show using OpenCV
	original_image = np.transpose(original_image, (1, 2, 0))
	original_image = np.ascontiguousarray(original_image)
	original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

	original_image = cv2.resize(original_image, (800, 800), interpolation=cv2.INTER_CUBIC)

	width, height, _ = original_image.shape

	# Process each pair of points as (x, y) coords
	for index in range(0, len(output), 2):
		x, y = int(output[index] * width), int(output[index + 1] * height)

		cv2.circle(original_image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

	# Show the resulting image
	cv2.imshow("test", original_image)
	cv2.waitKey(0)


if __name__ == "__main__":
	main()