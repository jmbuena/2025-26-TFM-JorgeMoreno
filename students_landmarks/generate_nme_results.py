import os
import subprocess
from glob import glob



def main():
    base_models_folder = 'data'

    cmd = 'python test/students_landmarks_database.py --anns-file wflw_ann_test.txt --database wflw --gpu 0 --regressor encoder --backbone {model} --batch-size 64 --epochs 100 --patience 20 --size {size} --save-file'

    for model_path in glob(os.path.join(base_models_folder, '**', 'best.ckpt'), recursive=True):
        parts = model_path.split('/')

        model_name = parts[3]
        model_size = parts[4]

        model_cmd = cmd.replace('{model}', model_name).replace('{size}', model_size)
        print(model_cmd)

        subprocess.run(model_cmd.split(' '))


if __name__ == '__main__':
    main()
