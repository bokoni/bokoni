/// <reference types="vite/client" />

declare var MonacoEnvironment: { getWorker(workerId: string, label: string): Worker } | undefined;
