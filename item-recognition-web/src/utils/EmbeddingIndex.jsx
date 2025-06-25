const EMBED_DIM = 576;
let embeddings = null;

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
