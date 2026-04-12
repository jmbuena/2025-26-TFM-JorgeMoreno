# List of commands that I normally use on the project

## Train a model
Set the command options as desired:

```bash
python test/students_landmarks_train.py --anns-file wflw_ann_train.txt --database wflw --gpu 0 --regressor encoder --backbone mobilenet_v3_small --batch-size 64 --epochs 100 --patience 20 --size 64
```

## Convert the model to ONNX format
```bash
python convert_to_onnx.py --anns-file wflw_ann_train.txt --database wflw --gpu 0 --regressor encoder --backbone resnet18 --batch-size 64 --epochs 100 --patience 20 --size 256 --output-name resnet18
```

## Extract a list of test images
```bash
python utils/extract_random_test_images.py --anns-file wflw_ann_test.txt --output-folder test_images/
```