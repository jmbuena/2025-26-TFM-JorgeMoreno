<script lang="ts">
    import type { ImageSize } from "../ai/AiHelpers";
    import { AnnotationRow, processCsvFile } from "../ai/Csv";
    import { Landmark } from "../ai/Landmarks";
    import { startPipeline, type OutputStats } from "../ai/Pipeline";
    import DynamicCanvas from "./DynamicCanvas.svelte";

	let images: Array<{ label: string, image: ImageData, size: ImageSize }> = $state([]);
	let realLandmarks: Array<Landmark> = $state([]);
	let outputLandmarks: Array<Landmark> = $state([]);
	let landmarkNumbers: Array<unknown> = $state([]);
	let stats: OutputStats | undefined = $state(undefined);

	let annotations: Map<string, AnnotationRow> = $state(new Map());

	export function showCanvas(label: string, image: ImageData, size: ImageSize): void {
		images.push({ label, image, size });
	}

	export function displayTable(
		newOutputLandmarks: Array<Landmark>,
		newRealLandmarks: Array<Landmark> | undefined,
		newStats: OutputStats | undefined,
	): void {
		outputLandmarks = newOutputLandmarks;
		realLandmarks = newRealLandmarks ?? [];
		stats = newStats;

		landmarkNumbers = new Array(outputLandmarks.length);
	}

	async function start(event: Event): Promise<void> {
		const files: Array<File> = (event.target! as any).files as Array<File>;
		const file = files[0];

		await startPipeline(file, annotations, showCanvas, displayTable);
	}

	async function readAnnotations(event: Event): Promise<void> {
		const files: Array<File> = (event.target! as any).files as Array<File>;
		const file = files[0];

		annotations = processCsvFile(await file.text());
	}
</script>

<div class="flex justify-center">
	<input type="file" accept="image/*" onchange={start} class="px-2 py-1 border rounded">

	<input type="file" accept=".csv,.txt" onchange={readAnnotations} class="px-2 py-1 border rounded">
</div>

<div class="flex flex-wrap gap-5 p-5">
	{#each images as data}
		<div class="">
			<p class="text-orange-500 font-semibold">{data.label}</p>
			<DynamicCanvas imageData={data.image} size={data.size} />
		</div>
	{/each}

	{#if outputLandmarks.length > 0}
		<div>
			<h2 class="text-xl font-semibold">Data</h2>

			<div class="mb-2 space-y-1">
				<p>Median: {stats?.medianDistance.toFixed(4)}</p>
				<p>Deviation: {stats?.deviation.toFixed(4)}</p>
			</div>

			<table class="border-collapse border border-gray-400">
				<thead>
					<tr>
						<th class="p-2 border border-gray-300">Output</th>
						<th class="p-2 border border-gray-300">Expected</th>
						<th class="p-2 border border-gray-300">Euclidean Distance</th>
					</tr>
				</thead>
				<tbody>
					{#each outputLandmarks as _, index}
						<tr class="hover:bg-neutral-700">
							<td class="p-2 border border-gray-300">
								{outputLandmarks[index]}
							</td>
							<td class="p-2 border border-gray-300">
								{realLandmarks.at(index) ?? "-"}
							</td>
							<td class="p-2 border border-gray-300">
								{stats?.landmarkDistance.at(index)?.toFixed(4) ?? "-"}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
