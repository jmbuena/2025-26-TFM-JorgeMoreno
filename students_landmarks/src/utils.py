import os
from typing import Tuple

import numpy as np
from images_framework.src.datasets import Database

from src.dataloader import Mode, MyDataset
from src.students_landmarks import StudentsLandmarks



def parse_common_onnx_options() -> Tuple[list[str], dict[str, str]]:
	"""
	Parse options from command line.
	"""
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--anns-file',
		'-a',
		dest='anns_file',
		required=True,
		help='Ground truth annotations file.'
	)

	parser.add_argument(
		'--onnx-path',
		'-p',
		dest='onnx_path',
		required=True,
		help='Path to the either output or input ONNX model.'
	)

	args, unknown = parser.parse_known_args()

	return unknown, args



def load_first_annotation(anns_file):
	"""
	Load ONE ground truth annotations according to each database.
	"""
	print('Open annotations file: ' + str(anns_file))

	if os.path.isfile(anns_file):
		pos = anns_file.rfind('/') + 1
		path = anns_file[:pos]
		file = anns_file[pos:]

		db = file[:file.find('_ann')]
		datasets = [subclass().get_names() for subclass in Database.__subclasses__()]

		with open(anns_file, 'r', encoding='utf-8') as ifs:
			lines = ifs.readlines()
			anns = []

			for i in range(len(lines)):
				parts = lines[i].strip().split(';')

				if parts[0] == '@':
					db = parts[1]

				if parts[0] == '#' or parts[0] == '@':
					continue

				idx = next((idx for idx, subset in enumerate(datasets) if db in subset), None)

				if idx is None:
					raise ValueError('Database does not exist')
				seq = Database.__subclasses__()[idx]().load_filename(path, db, lines[i])

				if len(seq.images) == 0:
					continue

				anns.append(seq)

				break # DO NOT LOAD THE REST
		ifs.close()
	else:
		raise ValueError('Annotations file does not exist')

	return anns


def get_one_image_from_dataset(anns_file: str, studentLandmarks: StudentsLandmarks):
	anns = load_first_annotation(anns_file)
	dataset_train = MyDataset(anns, studentLandmarks.indices, studentLandmarks.regressor, studentLandmarks.width, studentLandmarks.height, Mode.TEST)

	img: np.ndarray = dataset_train[0]["img"]
	img = img.astype(np.float32)

	return img
