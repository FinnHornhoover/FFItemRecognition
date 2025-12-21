const EMBED_DIM = 576;
let crateEmbeddings = null;
let embeddings = null;
let crateEmbeddingsLoaded = false;
let embeddingsLoaded = false;

export async function loadCrateEmbeddings(embeddingsURL) {
    // Helper to fetch and read binary data
    async function fetchArrayBuffer(url) {
        const response = await fetch(url);
        return response.arrayBuffer();
    }

    // Load the file
    const embeddingsBuffer = await fetchArrayBuffer(embeddingsURL);
    const flatArray = new Float32Array(embeddingsBuffer);

    // Convert flat array to array of embeddings
    // Assuming each crate embedding is EMBED_DIM floats
    const numCrates = flatArray.length / EMBED_DIM;
    if (numCrates !== Math.floor(numCrates)) throw new Error("Invalid crate embeddings size");

    crateEmbeddings = new Array(numCrates);
    for (let i = 0; i < numCrates; i++) {
        crateEmbeddings[i] = flatArray.slice(i * EMBED_DIM, (i + 1) * EMBED_DIM);
    }
    crateEmbeddingsLoaded = true;
}

export async function loadQuantizedEmbeddings(qvalsURL, startsURL, stepsURL) {
    // Helper to fetch and read binary data
    async function fetchArrayBuffer(url) {
        const response = await fetch(url);
        return response.arrayBuffer();
    }

    // Load the files
    const [qvalsBuffer, startsBuffer, stepsBuffer] = await Promise.all([
        fetchArrayBuffer(qvalsURL),
        fetchArrayBuffer(startsURL),
        fetchArrayBuffer(stepsURL)
    ]);

    const starts = new Float32Array(startsBuffer); // length 576
    const steps = new Float32Array(stepsBuffer);   // length 576
    const qvals = new Uint8Array(qvalsBuffer);     // length = n * 576

    const n = qvals.length / EMBED_DIM;
    if (n !== Math.floor(n)) throw new Error("Invalid qvals size");

    // Reconstruct: output as Float32Array[n][576]
    embeddings = new Array(n);
    for (let j = 0; j < n; j++) {
        const vec = new Float32Array(EMBED_DIM);
        for (let i = 0; i < EMBED_DIM; i++) {
            vec[i] = qvals[j * EMBED_DIM + i] * steps[i] + starts[i];
        }
        embeddings[j] = vec;
    }
    embeddingsLoaded = true;
}

export function areCrateEmbeddingsLoaded() {
    return crateEmbeddings !== null && crateEmbeddings.length > 0;
}

export function areEmbeddingsLoaded() {
    return embeddings !== null && embeddings.length > 0;
}

function distance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

export function getNearestNeighbor(embedding) {
    let minDistance = Infinity;
    let minIndex = -1;
    for (let i = 0; i < embeddings.length; i++) {
        const d = distance(embedding, embeddings[i]);
        if (d < minDistance) {
            minDistance = d;
            minIndex = i;
        }
    }
    return {index: minIndex, distance: minDistance};
}

export function getNearestCrateNeighbor(embedding) {
    if (!crateEmbeddings || crateEmbeddings.length === 0) {
        throw new Error("Crate embeddings not loaded yet");
    }
    let minDistance = Infinity;
    let minIndex = -1;
    for (let i = 0; i < crateEmbeddings.length; i++) {
        const d = distance(embedding, crateEmbeddings[i]);
        if (d < minDistance) {
            minDistance = d;
            minIndex = i;
        }
    }
    return {index: minIndex, distance: minDistance};
}

export function getTopNNeighbors(embedding, n = 10) {
    const distances = [];
    for (let i = 0; i < embeddings.length; i++) {
        const d = distance(embedding, embeddings[i]);
        distances.push({index: i, distance: d});
    }
    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, n);
}
