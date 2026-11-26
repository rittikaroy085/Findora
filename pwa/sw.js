
const CACHE_NAME = "findora-v1";

self.addEventListener("install", (event) => {
    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        self.clients.claim()
    );
});
