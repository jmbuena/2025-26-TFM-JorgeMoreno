<script lang="ts">
    import { objectToCsvString, processCsvFile } from "../ai/Csv";
    import { startPipeline, startPipelineWithPerfectAnnotations } from "../ai/Pipeline";
    import { ClassificationModel } from "../ai/RunModel";

	const models: Record<string, Record<string, string>> = {
		"ResNet50 (old)": {
			"256": "FaceMerged",
		},
		"ResNet50": {
			"256": "resnet50_256",
			"128": "resnet50_128",
			"64": "resnet50_64",
		},
		"MobileNet V3 Small": {
			"256": "mobilenet_v3_small_256",
			"128": "mobilenet_v3_small_128",
			"64": "mobilenet_v3_small_64",
		},
		"MobileNet V3 Large": {
			"256": "mobilenet_v3_large_256",
			"128": "mobilenet_v3_large_128",
			"64": "mobilenet_v3_large_64",
		},
		"ResNet18": {
			"256": "resnet18_256",
			"128": "resnet18_128",
			"64": "resnet18_64",
		},
		"EfficientNet-b0": {
			"256": "efficientnet-b0_256",
			"128": "efficientnet-b0_128",
			"64": "efficientnet-b0_64",
		}
	};

	let images = $state<Array<File>>([]);
	let annotationsFile: File | undefined = $state(undefined);
	let canStartProcessing = $state(false);
	let isProcessing = $state(false);
	let selectedModelName = $state(Object.keys(models)[0]);
	let selectedModelSize = $state(Object.keys(models[Object.keys(models)[0]])[0]);
	let tableData: Array<Record<string, number | string>> = $state([]);
	let tableDataStats: Record<string, number> = $state({});
	let warmupCount: number = $state(10);
	let useGPU: boolean = $state(true);
	let otherStats = $state({
		facesMissing: 0,
		totalTime: 0,
		warmupTime: 0,
		csvProcessingTime: 0,
	});

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

		const startProcessingCsv = performance.now();
		const annotations = annotationsFile
			? processCsvFile(await annotationsFile.text())
			: new Map();
		otherStats.csvProcessingTime = performance.now() - startProcessingCsv;

		const numberDurations: Array<Record<string, number>> = [];

		const model = new ClassificationModel();
		model.setUseGPU(useGPU);
		
		const modelPath = models[selectedModelName][selectedModelSize];
		await model.load(modelPath);

		// Warmup
		const startWarmup = performance.now();
		for (let i = 0; i < warmupCount; i++) {
			const image = images[i];
			await startPipelineWithPerfectAnnotations(modelPath, image, annotations, model, Number.parseInt(selectedModelSize));
		}

		otherStats.warmupTime = performance.now() - startWarmup;

		// Execution
		for (let i = warmupCount; i < images.length; i++) {
			const image = images[i];

			const startExecution = performance.now();
			
			const result = await startPipelineWithPerfectAnnotations(modelPath, image, annotations, model, Number.parseInt(selectedModelSize))
				.catch((error) => {
					console.error(error);
				});

			otherStats.totalTime += performance.now() - startExecution;

			if (result && "timings" in result) {
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
			} else {
				tableData.push({
					"Nombre imagen": image.name,
				});

				otherStats.facesMissing++;
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
			tableDataStats[key] = stat / (tableData.length - otherStats.facesMissing);
		}

		isProcessing = false;
	}

	function copyTableClipboardAsCvs(): void {
		const modelPath = models[selectedModelName][selectedModelSize];

		const tableDataWithExtraData = new Array(tableData.length);
		let index = 0;

		for (const row of tableData) {
			tableDataWithExtraData[index] = {
				...row,
				with_gpu: useGPU ? 1 : 0,
				model: modelPath,
				input_size: Number.parseInt(selectedModelSize),
			};

			index++;
		}

		const csvString = objectToCsvString(tableDataWithExtraData);
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
		<select id="modelNames" bind:value={selectedModelName} class="px-2 py-1 border rounded">
			{#each Object.keys(models) as modelName}
				<option value={modelName} class="text-black">{modelName}</option>
			{/each}
		</select>
	</div>

	<div class="flex items-center gap-x-2">
		<p>Tamaño input: </p>
		<select id="modelNames" bind:value={selectedModelSize} class="px-2 py-1 border rounded">
			{#each Object.keys(models[selectedModelName]) as modelSize}
				<option value={modelSize} class="text-black">{modelSize}</option>
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

	<div class="flex items-center gap-x-2">
		<p>Calentamiento: </p>
		<input
			type="number"
			id="calentamiento"
			min="0" max={images.length}
			bind:value={warmupCount}
			disabled={images.length === 0}
			class="px-2 py-1 border rounded hover:bg-neutral-700 not-disabled:hover:bg-neutral-700 disabled:border-neutral-700 disabled:text-neutral-700"
		>
	</div>

	<div class="flex items-center gap-x-2">
		<p>Usar GPU: </p>
		<input
			type="checkbox"
			id="useGPU"
			bind:checked={useGPU}
			class="px-2 py-1 border rounded hover:bg-neutral-700 not-disabled:hover:bg-neutral-700 disabled:border-neutral-700 disabled:text-neutral-700"
		>
	</div>

	<div>
		<button
			onclick={startProcessing}
			class="flex items-center px-2 py-1 border rounded not-disabled:hover:bg-neutral-700 disabled:border-neutral-700 disabled:text-neutral-700"
			disabled={!canStartProcessing}
		>
			{#if isProcessing}
				<svg xmlns="http://www.w3.org/2000/svg" width="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-1 animate-spin lucide lucide-loader-icon lucide-loader"><path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/><path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/><path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/><path d="m4.9 4.9 2.9 2.9"/></svg>
			{/if}
			<span>Procesar</span>
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

			<div>
				<h2 class="text-xl">Otras estadísticas:</h2>
				<p>Caras no encontradas: {otherStats.facesMissing}</p>
				<p>Tiempo lectura CSV: {(otherStats.csvProcessingTime / 1000).toFixed(4)}s</p>
				<p>Tiempo calentamiento: {(otherStats.warmupTime / 1000).toFixed(4)}s</p>
				<p>Tiempo procesamiento: {(otherStats.totalTime / 1000).toFixed(4)}s</p>
			</div>
		{:else}
			<p class="">No se ha procesado el modelo todavía</p>
		{/if}
	</div>
</div>
