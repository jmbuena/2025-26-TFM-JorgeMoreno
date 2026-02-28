<script lang="ts">
	let annotations: File;

	const hints = {
		filename: 1,
	};

	async function readAnnotations(event: Event): Promise<void> {
		const files: Array<File> = (event.target! as any).files as Array<File>;

		const file = files[0];
		const contents = await file.text();
		const lines = contents.split("\n");

		console.log(lines[0]);
	}
</script>

<div>
	<input type="file" accept=".csv,.txt" onchange={readAnnotations}>
</div>
