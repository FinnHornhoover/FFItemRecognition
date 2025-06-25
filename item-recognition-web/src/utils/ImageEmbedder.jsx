import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js';
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;

let session = null;

export async function loadModel(modelPath = '/embedder.onnx') {
  session = await ort.InferenceSession.create(modelPath);
}

export async function getEmbedding(imageTensors) {
  if (!session) throw new Error("Model not loaded yet");
  const { dataArray, dims } = imageTensors;

  const allEmbeddings = [];
  for (const data of dataArray) {
    const ortTensor = new ort.Tensor('float32', data, dims);
    const feeds = { input: ortTensor };
    const results = await session.run(feeds);
    const embeddings = results.output.data;
    allEmbeddings.push(embeddings);
  }
  return allEmbeddings;
}
