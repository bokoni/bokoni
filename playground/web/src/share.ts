// Code in a link: compressed (deflate) then base64url, so that a link stays short.

const toBase64Url = (bytes: Uint8Array) =>
  btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join(""))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");

const fromBase64Url = (s: string) =>
  Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));

async function through(bytes: Uint8Array, stream: CompressionStream | DecompressionStream) {
  const out = new Blob([bytes as BlobPart]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(out).arrayBuffer());
}

export async function encodeShared(code: string): Promise<string> {
  const bytes = new TextEncoder().encode(code);
  return toBase64Url(await through(bytes, new CompressionStream("deflate-raw")));
}

export async function decodeShared(s: string): Promise<string> {
  const bytes = await through(fromBase64Url(s), new DecompressionStream("deflate-raw"));
  return new TextDecoder().decode(bytes);
}
