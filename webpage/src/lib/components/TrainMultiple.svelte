<script lang="ts">
    import { objectToCsvString, processCsvFile } from "../ai/Csv";
    import { startPipeline } from "../ai/Pipeline";
    import { ClassificationModel } from "../ai/RunModel";

	const models = {
		"FaceMerged": "ResNet50 (old)",
		"mobilenet_v3_small": "MobileNet V3 Small",
	};

	let images = $state<Array<File>>([]);
	let annotationsFile: File | undefined = $state(undefined);
	let canStartProcessing = $state(false);
	let isProcessing = $state(false);
	let selectedModel = $state(Object.keys(models)[0]);
	let tableData: Array<Record<string, number | string>> = $state([]);
	let tableDataStats: Record<string, number> = $state({});

	function setImages(event: Event) {
		images = (event.target! as any).files as Array<File>;
	}

	function setAnnotationFile(event: Event) {
		annotationsFile = (event.target! as any).files[0] as File;
	}

	async function startProcessing() {
		isProcessing = true;

		tableDataStats = {};
		tableData = [];

		const annotations = annotationsFile
			? processCsvFile(await annotationsFile.text())
			: new Map();

		const numberDurations: Array<Record<string, number>> = [];

		const model = new ClassificationModel();

		for (const image of images) {
			const result = await startPipeline(selectedModel, image, annotations, model);

			if ("timings" in result) {
				const durations: Record<string, number> = {};
				
				const timings = result.timings.getTimings();
				for (const [key, timing] of Object.entries(timings)) {
					durations[key] = timing.duration();
				}
				
				numberDurations.push();

				tableData.push({
					"Nombre imagen": image.name,
					...durations,
					"Error medio": result.errors?.deviation ?? 0,
					"Desviación típica": result.errors?.medianDistance ?? 0,
				});
			}
		}

		// Calculate the total of the values
		for (const data of tableData) {
			for (const key in data) {
				if (typeof data[key] === "number") {
					if (key in tableDataStats) {
						tableDataStats[key] += data[key];
					} else {
						tableDataStats[key] = data[key];
					}
				}
			}

		}

		for (const [key, stat] of Object.entries(tableDataStats)) {
			tableDataStats[key] = stat / tableData.length;
		}

		isProcessing = false;
	}

	function copyTableClipboardAsCvs(): void {
		const csvString = objectToCsvString(tableData);
		navigator.clipboard.writeText(csvString);
	}

	$effect(() => {
		canStartProcessing = !isProcessing && images.length > 0;
	});
</script>


<div class="mx-10 space-y-1">
	<h2 class="text-xl">Datos entrada</h2>
	<div class="flex items-center gap-x-2">
		<p>Modelo: </p>
		<select id="modelNames" bind:value={selectedModel} class="px-2 py-1 border rounded">
			{#each Object.entries(models) as [modelPath, modelName]}
				<option value={modelPath} class="text-black">{modelName}</option>
			{/each}
		</select>
	</div>

	<div class="flex items-center gap-x-2">
		<p>Imágenes: </p>
		<label for="images" class="px-2 py-1 border rounded hover:bg-neutral-700">Seleccionar</label>
		<p class="{images.length > 0 ? 'text-green-500' : 'text-red-500'}">{images.length} imágenes seleccionadas</p>
		<input type="file" id="images" accept="image/*" multiple oninput={setImages} class="hidden">
	</div>

	<div class="flex items-center gap-x-2">
		<p>Anotaciones: </p>
		<label for="anns" class="px-2 py-1 border rounded hover:bg-neutral-700">Seleccionar</label>
		{#if annotationsFile}
			<p class="text-green-500">Anotaciones cargadas</p>
		{:else}
			<p class="text-red-500">Sin anotaciones</p>
		{/if}
		<input type="file" id="anns" accept=".txt" oninput={setAnnotationFile} class="hidden">
	</div>

	<div>
		<button
			onclick={startProcessing}
			class="px-2 py-1 border rounded not-disabled:hover:bg-neutral-700 disabled:border-neutral-700 disabled:text-neutral-700"
			disabled={!canStartProcessing}
		>
			Procesar
		</button>
	</div>

	<!-- Resultados -->
	<div class="mt-5">
		<h2 class="text-xl">Resultados</h2>
		{#if tableData.length > 0}
			<table class="table-auto rounded">
				<thead class="">
					<tr class="">
						<th class="pl-2 text-end border border-neutral-700 px-2">Índice</th>

						{#each Object.keys(tableData[0]) as key}
							<th class="pl-2 text-end border border-neutral-700 px-2">{key}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each tableData as row, index}
						<tr>
							<td class="text-end border border-neutral-700 px-2">{index}</td>

							{#each Object.values(row) as value}
								<td class="text-end border border-neutral-700 px-2">{typeof value === "number" ? value.toFixed(4) : value}</td>
							{/each}
						</tr>
					{/each}

					<tr>
						<td></td>
						<td></td>

						{#each Object.values(tableDataStats) as stat}
							<td class="px-2 text-orange-500 border border-neutral-700 text-end bg-neutral-950">{stat.toFixed(4)}</td>
						{/each}
					</tr>
				</tbody>
			</table>

			<div>
				<button class="my-3 px-2 py-1 border rounded hover:bg-neutral-700" onclick={copyTableClipboardAsCvs}>Guardar como CSV</button>
			</div>
		{:else}
			<p class="">No se ha procesado el modelo todavía</p>
		{/if}
	</div>
</div>
