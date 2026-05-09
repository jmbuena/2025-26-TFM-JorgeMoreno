import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt


def find_files(folder: str, pattern: list[str]) -> list[str]:
    return glob(os.path.join(folder, *pattern), recursive=True)


def _read_csv(filepaths: list[str]):
    df = pd.DataFrame()

    for filepath in filepaths:
        new_df = pd.read_csv(filepath, sep=',')
        df = pd.concat([df, new_df], ignore_index=True)

    df.columns = df.columns.str.strip()
    df['model'] = df['model'].str.strip()

    # debug_df = df.loc[df['input_size'] == 256]
    # debug_df = df.where(df['input_size'] == 256)
    # debug_df.to_csv('./debug.csv', sep=';')

    return df


def _read_stat_results(filepath: str):
    with open(filepath) as file:
        data = {}
        section = ''

        for line in file:
            line = line.strip()

            if line.startswith('>'):
                section = line.replace('> ', '').replace(':', '')
                data[section] = {}
            else:
                parts = line.split(':')
                data[section][parts[0]] = float(parts[1].strip())
    
    return {os.path.basename(filepath).replace('.txt', ''): data}


def _read_stats(files: list[str]) -> dict:
    data = {}

    for file in files:
        file_data = _read_stat_results(file)
        data |= file_data
    
    return data


def _extract_stats(model_stats: dict, timings: pd.DataFrame):
    data = dict()

    for model, stats in model_stats.items():
        nme = stats['Total']['NME']

        model_timings = timings.loc[timings['model'] == model]

        cpu_rows = model_timings.loc[model_timings['with_gpu'] == 0]['runmodel']
        cpu_mean = cpu_rows.mean()
        cpu_std = cpu_rows.std()

        gpu_rows = model_timings.loc[model_timings['with_gpu'] == 1]['runmodel']
        gpu_mean = gpu_rows.mean()
        gpu_std = gpu_rows.std()

        model_parts = model.split('_')
        model_name = '_'.join(model_parts[:-1])
        input = int(model_parts[-1])

        if model_name not in data:
            data[model_name] = {}

        data[model_name][input] = {
            'nme': nme,
            'cpu_mean': cpu_mean,
            'cpu_std': cpu_std,
            'gpu_mean': gpu_mean,
            'gpu_std': gpu_std,
        }

    # Sort data
    for model_name, inputs in data.items():
        data[model_name] = dict(sorted(inputs.items()))

    return data
        

def _plot(data: dict, draw_gpu: bool, draw_cpu: bool):
    fig, ax = plt.subplots()
    print(data)

    marker_sizes = {
        64: 10,
        128: 25,
        256: 50,
    }

    markers = ['.', '^', 's']

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0

    for model, inputs in data.items():
        cpu_points = {'x': [], 'y': [], 's': []}
        gpu_points = {'x': [], 'y': [], 's': []}

        for input, input_data in inputs.items():
            cpu_points['x'].append(input_data['cpu_mean'])
            cpu_points['y'].append(input_data['nme'])
            cpu_points['s'].append(marker_sizes[input])

            gpu_points['x'].append(input_data['gpu_mean'])
            gpu_points['y'].append(input_data['nme'])
            gpu_points['s'].append(marker_sizes[input])

        if draw_cpu:
            ax.plot(cpu_points['x'], cpu_points['y'], label=f'{model} (cpu)', color=colors[color_index])

            for index in range(len(inputs)):
                ax.scatter(cpu_points['x'][index], cpu_points['y'][index], color=colors[color_index], marker=markers[index])
            # ax.scatter(cpu_points['x'], cpu_points['y'], s=cpu_points['s'], color=colors[color_index])

        if draw_gpu:
            ax.plot(gpu_points['x'], gpu_points['y'], label=f'{model} (gpu)', linestyle='dashed', color=colors[color_index])

            for index in range(len(inputs)):
                ax.scatter(gpu_points['x'][index], gpu_points['y'][index], color=colors[color_index], marker=markers[index])

        ax.legend()

        color_index += 1

    plt.xlabel('Tiempo (ms)')
    plt.ylabel('NME')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=None)

    plt.show()


def main():
    files = find_files('./time_measurements/', ['**', '*.csv'])
    timings = _read_csv(files)

    files = find_files('./stat_results/', ['**', '*.txt'])
    stats = _read_stats(files)

    stats = _extract_stats(stats, timings)
    _plot(stats, draw_cpu=True, draw_gpu=True)


if __name__ == '__main__':
    main()

