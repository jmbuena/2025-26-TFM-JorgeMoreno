import { router } from "./src/routes.ts";


if (import.meta.main) {
	Deno.serve({
		hostname: "0.0.0.0",
		port: 8080,
	}, router.fetch);
}
