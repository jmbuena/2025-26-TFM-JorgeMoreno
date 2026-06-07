import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt

from src.utils import get_one_image_from_dataset
from images_framework.src.constants import Modes
from src.students_landmarks import StudentsLandmarks


IMG_INDEX = 100

common_settings = {
    "patience": 20,
    "batch-size": 1,
    "epochs": 100,
    "regressor": "encoder",
    "database": "wflw",
    "gpu": 0,
}

models = [
    {
        **common_settings,
        "backbone": "efficientnet-b0",
        "size": 64,
        "color": (1, 0, 0),
    },
    {
        **common_settings,
        "backbone": "efficientnet-b0",
        "size": 128,
        "color": (0, 1, 0),
    },
    {
        **common_settings,
        "backbone": "efficientnet-b0",
        "size": 256,
        "color": (0, 0, 1),
    },
    # {
    #     **common_settings,
    #     "backbone": "mobilenet_v3_small",
    #     "size": 256,
    #     "color": (1, 0, 0),
    # },
]

fix_model_names = {
    "mobilenet_v3_small": "MobileNet V3 Small",
    "mobilenet_v3_large": "MobileNet V3 Large",
    "resnet50": "ResNet50",
    "resnet18": "ResNet18",
    "efficientnet-b0": "EfficientNet B0",
}

markers = {
    64: '.',
    128: '^',
    256: 's',
}

def main():
    # composite = Composite()
    # loaded_models = []

    # for model_args in models:
    #     args_ready: list[str] = []

    #     for key, value in model_args.items():
    #         args_ready.extend([f"--{key}", str(value)])

    #     print(args_ready)
        
    #     # Load the model
    #     sa = StudentsLandmarks('')
    #     sa.parse_options(args_ready)
    #     sa.load(Modes.TEST)
        
    #     composite.add(sa)
    #     loaded_models.append(sa.model)

    # composite.load(Modes.TEST)

    outputs = []
    higher_res_image = None

    for model_args in models:
        args_ready: list[str] = []

        for key, value in model_args.items():
            args_ready.extend([f"--{key}", str(value)])
    
        sa = StudentsLandmarks('')
        sa.parse_options(args_ready)
        sa.load(Modes.TEST)

        # Get one image to test the dataset
        img = get_one_image_from_dataset(anns_file="wflw_ann_train.txt", studentLandmarks=sa, img_index=IMG_INDEX)
        original_image = img.copy()

        if higher_res_image is None or original_image.size > higher_res_image.size:
            higher_res_image = original_image

        # Add the batch dimension to the image
        img = np.expand_dims(img, axis=0)

        # Start & run the inference
        backbone = model_args["backbone"]
        size = model_args["size"]
        ort_session = onnxruntime.InferenceSession(f'onnx_models/{backbone}_{size}.onnx')
        
        ort_outs = ort_session.run(None, {
            "x": img,
        })

        output = ort_outs[0][0]
        outputs.append({
            "backbone": backbone,
            "size": size,
            "output": output,
            "color": model_args["color"]
        })
    
    higher_res_image = _fix_and_show_image(higher_res_image)

    fig, ax = plt.subplots()
    ax.imshow(higher_res_image)

    width, height, _ = higher_res_image.shape

    # Process each pair of points as (x, y) coords
    for output_data in outputs:
        xs = []
        ys = []

        output = output_data["output"]

        for index in range(0, len(output), 2):
            x, y = int(output[index] * width), int(output[index + 1] * height)

            xs.append(x)
            ys.append(y)

        colors = [output_data["color"]] * len(xs)
        
        a = ax.scatter(xs, ys, c=colors, s=2, marker=markers[output_data["size"]])

        display_backbone = fix_model_names[output_data["backbone"]]
        a.set_label(f"{display_backbone} ({output_data["size"]})")
    
    ax.legend()
    
    plt.axis('off')
    plt.show()

    # for output in outputs:
    #     _draw_points(higher_res_image, output["output"], output["color"])
    
    # cv2.imshow("test", higher_res_image)
    # cv2.waitKey(0)



def _fix_and_show_image(original_image):
    # Fix the original image to show using OpenCV
    original_image = np.transpose(original_image, (1, 2, 0))
    original_image = np.ascontiguousarray(original_image)
    original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # original_image = cv2.resize(original_image, (800, 800), interpolation=cv2.INTER_CUBIC)

    return original_image

    # # Show the resulting image
    # cv2.imshow("test", original_image)
    # cv2.waitKey(0)


def _draw_points(original_image, output, color):
    width, height, _ = original_image.shape

    # Process each pair of points as (x, y) coords
    for index in range(0, len(output), 2):
        x, y = int(output[index] * width), int(output[index + 1] * height)

        cv2.circle(original_image, (x, y), radius=2, color=color, thickness=-1)


if __name__ == "__main__":
    main()
