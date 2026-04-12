import csv
import os
import random
import shutil


def parse_script_args() -> tuple[list[str], dict[str, str]]:
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
        help='Test annotations file.'
    )

    parser.add_argument(
        '--output-folder',
        dest='output_folder',
        required=True,
        help='Folder where to place all the test images in',
    )

    parser.add_argument(
        '--count',
        dest='count',
        required=False,
        help='How many images to grab',
        default=110,
    )

    args, unknown = parser.parse_known_args()

    return unknown, args



def main():
    _, args = parse_script_args()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    with open(args.anns_file, newline='') as csvfile:
        anns = csv.reader(csvfile, delimiter=',')

        # Skip the headers
        anns.__next__()

        image_paths = []

        for row in anns:
            image_paths.append(row[0])

        selected_image_paths = random.sample(image_paths, args.count)

        for image_path in selected_image_paths:
            shutil.copy(image_path, args.output_folder)


if __name__ == '__main__':
    main()
