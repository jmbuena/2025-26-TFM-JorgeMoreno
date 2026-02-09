<script>
	import { onMount } from "svelte";
	import { prepareModels } from "./lib/ai/LoadModel";
	import Canvas from "./lib/components/Canvas.svelte";
	import OpenCV from "./lib/components/OpenCV.svelte";
	import "/src/assets/ort-wasm-simd-threaded.jsep.mjs";

	let loading = $state(true);

	onMount(async () => {
		await prepareModels();
		loading = false;
	});
</script>

<OpenCV />

<div class="min-h-[100vh] py-10 flex items-center justify-center text-white">
	{#if loading}
		Loading...
	{:else}
		<Canvas />
	{/if}
</div>
