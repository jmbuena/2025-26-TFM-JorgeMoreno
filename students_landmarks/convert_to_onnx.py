import os
import sys
from typing import Tuple

import torch
from tqdm import tqdm
from images_framework.src.constants import Modes
from images_framework.src.datasets import Database

from src.dataloader import Mode, MyDataset
from src.students_landmarks import StudentsLandmarks


def parse_options() -> Tuple[list[str], dict[str, str]]:
	"""
	Parse options from command line.
	"""
	import argparse

	parser = argparse.ArgumentParser()
	# parser.add_argument(
	# 	'--model-path',
	# 	'-m',
	# 	dest='model_path',
	# 	required=True,
	# 	help='Trained model filepath (.ckpt).',
	# )
	parser.add_argument(
		'--anns-file',
		'-a',
		dest='anns_file',
		required=True,
		help='Ground truth annotations file.'
	)

	# parser.add_argument(
	# 	'--image-path',
	# 	'-i',
	# 	dest='image_path',
	# 	required=True,
	# 	help='Sample train image path.',
	# )

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
            for i in tqdm(range(len(lines)), file=sys.stdout):
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


def main():
	unknown, args = parse_options()
	# image_path = args.image_path
	anns_file = args.anns_file

	# image = cv2.imread(image_path)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# image = image.astype("float32") / 255.0
	# tensor = torch.from_numpy(image)
	# tensor = tensor.permute(2, 0, 1)
	# imageTensor = tensor.unsqueeze(0)

	# print(imageTensor.size())
	# input("asdads")

	sa = StudentsLandmarks('')
	sa.parse_options(unknown)
	sa.load(Modes.TEST)

	anns = load_first_annotation(anns_file)
	dataset_train = MyDataset(anns, sa.indices, sa.regressor, sa.width, sa.height, Mode.TEST)

	img = dataset_train[0]["img"]
	tensor = torch.tensor(img, dtype=torch.float32)
	imageTensor = tensor.unsqueeze(0)

	# print(img.shape)
	# input("Confirm continue?")

	onnx_program = sa.model.to_onnx(
		"output.onnx",
		(imageTensor,),
		export_params=True,
		# dynamo=True,
		# input_names=["image"],
		# output_names=["predictions"]
    )

	onnx_program.save("ModelMerged.onnx")

	# model = resnet50(weights=None)
	# model.load_state_dict(torch.load(model_path, weights_only=True)['state'])
	# model.eval()

	# image = cv2.imread(image_path)
	# img = transforms.ToTensor()(image)
	# example_inputs = img.unsqueeze(0)

	# print("Starting conversion...")

	# onnx_program = torch.onnx.export(model, example_inputs, f="outputs/ResNet/ResNet.onnx", dynamo=True, input_names=["image"], output_names=["predictions"])
	# onnx_program.save("ResNetMerged.onnx")

	# print("Conversion ended. The ONNX model is ready!")

	# onnx_model = torch.onnx.load("outputs/ResNet/ResNet.onnx", load_external_data=True)
	# torch.onnx.save(onnx_model, "resnet_merged.onnx")


if __name__ == "__main__":
	main()