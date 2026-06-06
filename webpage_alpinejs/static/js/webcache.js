const CACHE_NAME = "models-v1";

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  if (url.pathname.endsWith(".onnx") || url.pathname.endsWith("face-alignment.mjs")) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        const cached = await cache.match(event.request);

        if (cached) {
          console.log("Serving model from cache");
          return cached;
        }

        console.log("Downloading model");

        const response = await fetch(event.request);

        if (response.ok) {
          cache.put(event.request, response.clone());
        }

        return response;
      }),
    );
  }
});
