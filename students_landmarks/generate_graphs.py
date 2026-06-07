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
    
    return {os.path.basename(filepath).replace('.txt', '').replace('nme_', ''): data}


def _read_stats(files: list[str]) -> dict:
    data = {}

    for file in files:
        file_data = _read_stat_results(file)
        data |= file_data

    data = dict(sorted(data.items()))
    
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
        

def _plot(data: dict, draw_gpu: bool, draw_cpu: bool, target_model: str | None = None):
    fig, ax = plt.subplots()

    marker_sizes = {
        64: 10,
        128: 25,
        256: 50,
    }

    markers = ['.', '^', 's']

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0

    for model, inputs in data.items():
        if target_model is not None and not model.startswith(target_model):
            color_index += 1
            continue

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

        if draw_gpu:
            ax.plot(gpu_points['x'], gpu_points['y'], label=f'{model} (gpu)', linestyle='dashed', color=colors[color_index])

            for index in range(len(inputs)):
                ax.scatter(gpu_points['x'][index], gpu_points['y'][index], color=colors[color_index], marker=markers[index])

        ax.legend()

        color_index += 1

    plt.xlabel('Tiempo (ms)')
    plt.ylabel('NME')

    # ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0, top=None)

    plt.show()

def _print_as_latex_table(stats: dict):
    cpu_table = []
    gpu_table = []

    for model, sizes in stats.items():
        model = model.replace('_', '\\_')

        for size, data in sizes.items():
            cpu = f"{model} ({size}) & {data["nme"]:.4f} & {data["cpu_mean"]:.4f} & {data["cpu_std"]:.4f} \\\\"

            cpu_table.append(cpu)
            cpu_table.append("\\hline")

            gpu = f"{model} ({size}) & {data["nme"]:.4f} & {data["gpu_mean"]:.4f} & {data["gpu_std"]:.4f} \\\\"
            gpu_table.append(gpu)
            gpu_table.append("\\hline")
        
        cpu_table.append("\\hline")
        gpu_table.append("\\hline")
    
    print("\n".join(gpu_table))

    gains = {}
    for model, sizes in stats.items():
        for size, data in sizes.items():
            gains[f"{model} ({size})"] = {
                "absolute": data["cpu_mean"] - data["gpu_mean"],
                "relative": data["cpu_mean"] / data["gpu_mean"],
            }

    gains_table = []
    # for model, gain in gains.items():
    #     model = model.replace('_', '\\_')

    #     row = f"{model} & {gain["absolute"]:.4f} & {gain["relative"]:.3f} \\\\"
    #     print(row)
    #     print("\\hline")


def _draw_stds(stats: dict, draw_cpu: bool, draw_gpu: bool, target_model: str | None = None):
    marker_sizes = {
        64: 10,
        128: 25,
        256: 50,
    }

    markers = ['.', '^', 's']

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0

    fig, ax = plt.subplots()

    for model, inputs in stats.items():
        if target_model is not None and not model.startswith(target_model):
            color_index += 1
            continue

        cpu_points = {'x': [], 'y': [], 's': [], 'std': []}
        gpu_points = {'x': [], 'y': [], 's': [], 'std': []}

        for input, input_data in inputs.items():
            cpu_points['x'].append(input_data['cpu_mean'])
            cpu_points['y'].append(input_data['nme'])
            cpu_points['std'].append(input_data['cpu_std'])
            cpu_points['s'].append(marker_sizes[input])

            gpu_points['x'].append(input_data['gpu_mean'])
            gpu_points['y'].append(input_data['nme'])
            gpu_points['std'].append(input_data['gpu_std'])
            gpu_points['s'].append(marker_sizes[input])

        if draw_cpu:
            ax.errorbar(cpu_points['x'], cpu_points['y'], xerr=cpu_points['std'], label=f'{model} (cpu)', color=colors[color_index], capsize=3)

            for index in range(len(inputs)):
                ax.scatter(cpu_points['x'][index], cpu_points['y'][index], color=colors[color_index], marker=markers[index], s=20)
        
        if draw_gpu:
            ax.errorbar(gpu_points['x'], gpu_points['y'], xerr=gpu_points['std'], label=f'{model} (gpu)', linestyle='dashed', color=colors[color_index], capsize=3)

            for index in range(len(inputs)):
                ax.scatter(gpu_points['x'][index], gpu_points['y'][index],  color=colors[color_index], marker=markers[index], s=20)

        ax.legend()

        color_index += 1
    
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('NME')

    plt.show()


def main(draw_cpu: bool, draw_gpu: bool, filter_models: str | None):
    files = find_files('./time_measurements/', ['**', '*.csv'])
    timings = _read_csv(files)

    files = find_files('./stat_results/', ['**', '*.txt'])
    stats = _read_stats(files)

    stats = _extract_stats(stats, timings)

    # _plot(stats, draw_cpu=draw_cpu, draw_gpu=draw_gpu, target_model=filter_models)
    _draw_stds(stats, draw_cpu=draw_cpu, draw_gpu=draw_gpu, target_model=filter_models)
    # _print_as_latex_table(stats)
    


if __name__ == '__main__':
    main(draw_cpu=True, draw_gpu=True, filter_models='mobilenet')
