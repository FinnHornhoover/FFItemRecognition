import { createWorker, PSM } from 'tesseract.js';
import { fromPriceString } from './PriceConversion';

async function findSquares(src) {
  let img = src.clone();

  // Split channels
  let channels = new cv.MatVector();
  cv.split(img, channels);
  let blue = channels.get(0);
  let green = channels.get(1);
  let red = channels.get(2);

  // Create mask for really black pixels
  let mask = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
  let low = new cv.Mat(src.rows, src.cols, cv.CV_8UC1, [0, 0, 0, 0]);
  let high = new cv.Mat(src.rows, src.cols, cv.CV_8UC1, [10, 10, 10, 10]);

  let blueMask = new cv.Mat();
  let greenMask = new cv.Mat();
  let redMask = new cv.Mat();

  cv.inRange(blue, low, high, blueMask);
  cv.inRange(green, low, high, greenMask);
  cv.inRange(red, low, high, redMask);

  cv.bitwise_and(blueMask, greenMask, mask);
  cv.bitwise_and(mask, redMask, mask);

  // Turn the edge 2px of the mask into 255
  for (let i = 0; i < mask.rows; i++) {
    for (let j = 0; j < mask.cols; j++) {
      if (i < 2 || i > mask.rows - 2 || j < 2 || j > mask.cols - 2) {
        mask.data[i * mask.cols + j] = 255;
      }
    }
  }

  // Invert and apply morphological operations
  cv.bitwise_not(mask, mask);
  let kernel = cv.Mat.ones(3, 3, cv.CV_8UC1);
  cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel);

  // Find connected components
  let labels = new cv.Mat();
  let stats = new cv.Mat();
  let centroids = new cv.Mat();
  cv.connectedComponentsWithStats(mask, labels, stats, centroids);

  let potentialSquares = [];

  for (let i = 0; i < stats.rows; i++) {
    const x = stats.data32S[i * stats.cols + 0];
    const y = stats.data32S[i * stats.cols + 1];
    const width = stats.data32S[i * stats.cols + 2];
    const height = stats.data32S[i * stats.cols + 3];

    if (width > 192 || height > 192 || width < 24 || height < 24) {
      continue;
    }

    const aspectRatio = width / height;
    if (aspectRatio >= 0.5 && aspectRatio <= 2.0) {
      potentialSquares.push({x, y, width, height});
    }
  }

  // Deduplicate if a square is inside another square
  potentialSquares.sort((a, b) => (a.y !== b.y) ? a.y - b.y : a.x - b.x);

  let keepSquares = potentialSquares.map(() => true);

  for (let i = 0; i < potentialSquares.length; i++) {
    const s1 = potentialSquares[i];
    for (let j = 0; j < potentialSquares.length; j++) {
      const s2 = potentialSquares[j];
      if (
        i !== j
        && s1.x > s2.x
        && s1.y > s2.y
        && s1.x + s1.width < s2.x + s2.width
        && s1.y + s1.height < s2.y + s2.height
      ) {
        keepSquares[i] = false;
      }
    }
  }

  const filteredSquares = potentialSquares.filter((_, i) => keepSquares[i]);

  const worker = await createWorker('eng');
  await worker.setParameters({
    tessedit_pageseg_mode: PSM.SINGLE_LINE,
    tessedit_char_whitelist: '0123456789xkMm .',
  });

  let squares = [];

  for (const square of filteredSquares) {
    if (square.y + 5 * square.height / 4 > src.rows) {
      continue;
    }

    let extraInfo = {};

    let roi = src.roi({x: square.x, y: square.y + square.height, width: square.width, height: square.height / 4});
    let dsize = new cv.Size(square.width * 4, square.height);
    cv.resize(roi, roi, dsize, 0, 0, cv.INTER_LINEAR);
    cv.cvtColor(roi, roi, cv.COLOR_BGR2GRAY);
    let canvas = document.createElement('canvas');
    cv.imshow(canvas, roi);
    roi.delete();

    try {
      const { data: { text }} = await worker.recognize(canvas);

      const xSplit = text.split('x');
      if (xSplit.length > 1) {
        const quantity = xSplit[0].trim();
        extraInfo.quantity = parseInt(quantity);
      }

      if (text.toLowerCase().includes('k')) {
        const price = xSplit[xSplit.length - 1].toLowerCase().split('k')[0].trim();
        extraInfo.price = fromPriceString(price + 'k');
      }

      if (text.toLowerCase().includes('m')) {
        const price = xSplit[xSplit.length - 1].toLowerCase().split('m')[0].trim();
        extraInfo.price = fromPriceString(price + 'M');
      }
    } catch (e) {
      console.log(e);
    }

    squares.push({...square, extraInfo});
  }

  await worker.terminate();

  // Cleanup
  blue.delete(); green.delete(); red.delete();
  blueMask.delete(); greenMask.delete(); redMask.delete();
  low.delete(); high.delete();
  mask.delete(); labels.delete(); stats.delete(); centroids.delete(); kernel.delete();
  channels.delete(); img.delete();

  return squares;
}

function preprocessForONNX(src, rect) {
  // 1. Crop region
  const { x, y, width, height } = rect;
  let roi = src.roi({ x, y, width, height });

  // 2. Resize to 224x224
  let dsize = new cv.Size(224, 224);
  cv.resize(roi, roi, dsize, 0, 0, cv.INTER_LINEAR);

  // 3. Convert from BGR to RGB
  cv.cvtColor(roi, roi, cv.COLOR_BGR2RGB);

  // 4. Prepare Float32Array in NCHW format → [1, 3, 224, 224]
  const numPixels = 224 * 224;
  const floatArray = new Float32Array(3 * numPixels); // [3, 224, 224]
  const data = roi.data; // Uint8ClampedArray, [R, G, B, R, G, B, ...]

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < numPixels; i++) {
      floatArray[0 * numPixels + i] = (data[i * 3 + 0] / 255.0 - mean[0]) / std[0]; // R
      floatArray[1 * numPixels + i] = (data[i * 3 + 1] / 255.0 - mean[1]) / std[1]; // G
      floatArray[2 * numPixels + i] = (data[i * 3 + 2] / 255.0 - mean[2]) / std[2]; // B
  }

  roi.delete();
  return floatArray;
}

export async function preprocessImage(file) {
  cv = (cv instanceof Promise) ? await cv : cv;
  return new Promise((resolve) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
      const src = cv.imread(img);
      cv.cvtColor(src, src, cv.COLOR_RGBA2BGR);
      const squares = await findSquares(src);

      let tensors = [];
      let extraInfoArray = [];

      for (const square of squares) {
        const floatData = preprocessForONNX(src, square);
        tensors.push(floatData);
        extraInfoArray.push(square.extraInfo);
      }

      resolve({
        dataArray: tensors,
        dims: [1, 3, 224, 224], // ONNX expects [N, C, H, W]
        extraInfoArray,
      });
    };
  });
}
