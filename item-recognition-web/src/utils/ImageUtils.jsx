import { createWorker, PSM } from 'tesseract.js';
import { fromPriceString } from './PriceConversion';
import { readPNGMetadata } from './PNGMetadata';

const MAX_WIDTH = 4096;
const MAX_HEIGHT = 4096;

function lineLength(coords) {
  return Math.sqrt((coords[0] - coords[2]) ** 2 + (coords[1] - coords[3]) ** 2);
}

function lineAngle(coords) {
  return Math.atan2(coords[3] - coords[1], coords[2] - coords[0]) * 180 / Math.PI;
}

function skeletonize(src) {
  let srcCopy = src.clone();
  let size = new cv.Size(srcCopy.cols, srcCopy.rows);
  let skel = cv.Mat.zeros(size, cv.CV_8UC1);

  let element = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(3, 3));
  let done = false;

  while (!done) {
    let eroded = cv.Mat.zeros(size, cv.CV_8UC1);
    let temp = cv.Mat.zeros(size, cv.CV_8UC1);

    cv.erode(srcCopy, eroded, element);
    cv.dilate(eroded, temp, element);
    cv.subtract(srcCopy, temp, temp);
    cv.bitwise_or(skel, temp, skel);
    srcCopy = eroded.clone();

    let zeros = size.width * size.height - cv.countNonZero(srcCopy);
    if (zeros === size.width * size.height) {
      done = true;
    }

    eroded.delete();
    temp.delete();
  }

  // Cleanup
  srcCopy.delete();
  element.delete();

  return skel;
}

function extendLine(line, rows, cols) {
  let [x1, y1, x2, y2] = line;

  if (x1 === x2) {
    return [x1, 0, x2, rows - 1];
  }

  if (y1 === y2) {
    return [0, y1, cols - 1, y2];
  }

  let m = (y2 - y1) / (x2 - x1);
  let b = y1 - m * x1;

  let x_min = 0;
  let x_max = cols - 1;
  let y_min = 0;
  let y_max = rows - 1;

  let x_min_2 = (y_min - b) / m;
  let x_max_2 = (y_max - b) / m;

  let x_min_3 = Math.max(x_min, x_min_2);
  let x_max_3 = Math.min(x_max, x_max_2);
  let y_min_3 = Math.round(m * x_min_3 + b);
  let y_max_3 = Math.round(m * x_max_3 + b);

  return [x_min_3, y_min_3, x_max_3, y_max_3];
}

function clusterLinesByEndpoints(lines, rows, cols, distanceThreshold = 20.0) {
  let lineListExtended = lines.map((line) => [extendLine(line, rows, cols), lineLength(line)]);
  let clusters = [];

  for (const [line, lineLen] of lineListExtended) {
    let [x1, y1, x2, y2] = line;

    let found = false;

    for (const cluster of clusters) {
      let [x1c, y1c, x2c, y2c] = cluster.mean;

      if (lineLength([x1c, y1c, x1, y1]) < distanceThreshold && lineLength([x2c, y2c, x2, y2]) < distanceThreshold) {
        cluster.line.push(line);
        cluster.lineLengths.push(lineLen);

        let lineAndLengths = cluster.line.map((line, index) => [line, cluster.lineLengths[index]]);
        const lineLengthsSum = cluster.lineLengths.reduce((acc, length) => acc + length, 0);

        let x1_mean = lineAndLengths.reduce((acc, [ln, length]) => acc + ln[0] * length, 0) / lineLengthsSum;
        let y1_mean = lineAndLengths.reduce((acc, [ln, length]) => acc + ln[1] * length, 0) / lineLengthsSum;
        let x2_mean = lineAndLengths.reduce((acc, [ln, length]) => acc + ln[2] * length, 0) / lineLengthsSum;
        let y2_mean = lineAndLengths.reduce((acc, [ln, length]) => acc + ln[3] * length, 0) / lineLengthsSum;

        cluster.mean = [x1_mean, y1_mean, x2_mean, y2_mean];
        cluster.meanAngle = cluster.meanAngle;

        found = true;
        break;
      }
    }

    if (!found) {
      clusters.push({
        mean: [x1, y1, x2, y2],
        meanAngle: lineAngle([x1, y1, x2, y2]),
        line: [line],
        lineLengths: [lineLen],
      });
    }
  }

  return clusters;
}

function selectClustersByAngle(clusters, angleThreshold = 10.0, clusterPickThreshold = 45.0) {
  let angleClusters = [];

  for (let j = 0; j < clusters.length; j++) {
    const cluster = clusters[j];
    let found = false;

    for (const existingCluster of angleClusters) {
      if (Math.abs(cluster.meanAngle - existingCluster.meanAngle) < angleThreshold) {
        existingCluster.meanAngle = (existingCluster.meanAngle * existingCluster.indices.length + cluster.meanAngle) / (existingCluster.indices.length + 1);
        existingCluster.indices.push(j);

        found = true;
        break;
      }
    }

    if (!found) {
      angleClusters.push({
        meanAngle: cluster.meanAngle,
        indices: [j],
      });
    }
  }

  angleClusters.sort((a, b) => b.indices.length - a.indices.length);

  let selectedClusters = [angleClusters[0]];
  for (const cluster of angleClusters.slice(1)) {
    if (
      Math.abs(cluster.meanAngle - selectedClusters[0].meanAngle) > clusterPickThreshold &&
      Math.abs(Math.abs(cluster.meanAngle - selectedClusters[0].meanAngle) - 180) > clusterPickThreshold
    ) {
      selectedClusters.push(cluster);
      break;
    }
  }

  selectedClusters.sort((a, b) => Math.min(a.meanAngle, 180 - a.meanAngle) - Math.min(b.meanAngle, 180 - b.meanAngle));

  return selectedClusters.map((angleCluster) => angleCluster.indices.map((i) => clusters[i]));
}

function extractPotentialGridLines(src, houghThreshold = 200) {
  let srcGray = new cv.Mat();
  cv.cvtColor(src, srcGray, cv.COLOR_BGR2GRAY);
  cv.GaussianBlur(srcGray, srcGray, new cv.Size(5, 5), 0);
  cv.adaptiveThreshold(srcGray, srcGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2);
  let skel = skeletonize(srcGray);

  let linesMat = new cv.Mat();
  cv.HoughLinesP(srcGray, linesMat, 1, Math.PI / 180, houghThreshold, 32, 20);

  if (linesMat.rows === 0) {
    return [];
  }

  let lines = [];
  // Access lines
  for (let i = 0; i < linesMat.rows; ++i) {
    let pt = linesMat.data32S.subarray(i * 4, i * 4 + 4);
    lines.push([pt[0], pt[1], pt[2], pt[3]]);
  }

  let dominantLineClusters = clusterLinesByEndpoints(lines, src.rows, src.cols, 20.0);
  let clusterLists = selectClustersByAngle(dominantLineClusters, 10.0, 45.0);

  // Cleanup
  skel.delete();
  srcGray.delete();
  linesMat.delete();

  return clusterLists;
}

function toHomogeneous(p) {
  return [p[0], p[1], 1.0];
}

function cross(p1, p2) {
  return [
    p1[1] * p2[2] - p1[2] * p2[1],
    p1[2] * p2[0] - p1[0] * p2[2],
    p1[0] * p2[1] - p1[1] * p2[0],
  ];
}

function lineFromPoints(p1, p2) {
  return cross(toHomogeneous(p1), toHomogeneous(p2));
}

function linesToHomogeneous(lines) {
  return lines.map((line) => lineFromPoints([line[0], line[1]], [line[2], line[3]]));
}

function ransacVanishingPoint(lines, numIters = 1000, threshold = 0.005) {
  let bestVp = null;
  let bestInliers = 0;

  if (lines.length < 2) {
    return null;
  }

  for (let i = 0; i < numIters; i++) {
    let randomLines = lines.sort(() => Math.random() - 0.5).slice(0, 2);
    let line1 = randomLines[0];
    let line2 = randomLines[1];
    let vp = cross(line1, line2);
    if (Math.abs(vp[2]) < 1e-6) {
      continue;
    }
    vp = [vp[0] / vp[2], vp[1] / vp[2], 1.0];

    let inliers = lines.filter((line) => {
      let distance = Math.abs(line[0] * vp[0] + line[1] * vp[1] + line[2]) / (
        Math.sqrt(vp[0] ** 2 + vp[1] ** 2 + 1) * Math.sqrt(line[0] ** 2 + line[1] ** 2)
      );
      return distance < threshold;
    });

    if (inliers.length > bestInliers) {
      bestInliers = inliers.length;
      bestVp = vp;
    }
  }

  return bestVp;
}

function estimateRectifyingHomography(vp1, vp2) {
  if (vp1 === null || vp2 === null || Math.abs(vp1[2]) < 1e-6 || Math.abs(vp2[2]) < 1e-6) {
    const identity = cv.Mat.eye(3, 3, cv.CV_64FC1);
    return identity;
  }

  // Normalize vanishing points
  vp1 = [vp1[0] / vp1[2], vp1[1] / vp1[2], 1.0];
  vp2 = [vp2[0] / vp2[2], vp2[1] / vp2[2], 1.0];

  let vp1_rect = new cv.Mat(3, 1, cv.CV_64FC1, vp1);
  let vp2_rect = new cv.Mat(3, 1, cv.CV_64FC1, vp2);

  // Step 1: Projective rectification
  let l_inf = cross(vp1, vp2);
  l_inf = [l_inf[0] / l_inf[2], l_inf[1] / l_inf[2], 1.0];

  let H_proj = cv.Mat.eye(3, 3, cv.CV_64FC1);
  H_proj.data64F[6] = l_inf[0];
  H_proj.data64F[7] = l_inf[1];
  H_proj.data64F[8] = l_inf[2];

  // Step 2: Apply H_proj to vanishing points
  vp1_rect = H_proj.mul(vp1_rect);
  vp2_rect = H_proj.mul(vp2_rect);

  // Step 3: Rotation to align vp1_rect with x-axis
  let vec = vp1_rect.slice([0, 1]);
  let angle = -Math.atan2(vec[1], vec[0]);
  angle = angle > 0 ? angle : angle + 2 * Math.PI;
  angle = angle % Math.PI;
  angle = angle < Math.PI / 2 ? angle : angle - Math.PI;

  let R = cv.Mat.eye(3, 3, cv.CV_64FC1);
  R.data64F[0] = Math.cos(angle);
  R.data64F[1] = -Math.sin(angle);
  R.data64F[4] = Math.sin(angle);
  R.data64F[5] = Math.cos(angle);

  let H_rot = R.mul(H_proj, 1.0);

  // Step 4: Apply rotation to vanishing points
  let vp1_r = R.mul(vp1_rect, 1.0);
  let vp2_r = R.mul(vp2_rect, 1.0);

  // Step 5: Affine rectification — solve for b_hor
  let x1 = vp1_r.data64F[0];
  let y1 = vp1_r.data64F[1];
  let x2 = vp2_r.data64F[0];
  let y2 = vp2_r.data64F[1];

  let A = x1 * x2;
  let B = x1 * y2 + x2 * y1;
  let C = x1 * x2 + y1 * y2;

  let b_hor = 0.0;

  if (Math.abs(A) < 1e-8) {
    if (Math.abs(B) < 1e-8) {
      b_hor = 0.0;
    } else {
      b_hor = -C / B;
    }
  } else {
    let discriminant = B ** 2 - 4 * A * C;
    if (discriminant < 0) {
      b_hor = 0.0;
    } else {
      let sqrt_disc = Math.sqrt(discriminant);
      let b1 = (-B + sqrt_disc) / (2 * A);
      let b2 = (-B - sqrt_disc) / (2 * A);
      b_hor = b1 < b2 ? b1 : b2;
    }
  }

  let H_aff = cv.Mat.eye(3, 3, cv.CV_64FC1);
  H_aff.data64F[2] = b_hor;

  let H_full = H_aff.mul(H_rot, 1.0);

  // Cleanup
  vp1_rect.delete();
  vp2_rect.delete();
  l_inf.delete();
  H_proj.delete();
  R.delete();
  vp1_r.delete();
  vp2_r.delete();
  H_aff.delete();
  H_rot.delete();

  return H_full;
}

function adjustHomographyToView(src, H) {
  let h = src.rows;
  let w = src.cols;

  let corners = new cv.Mat(4, 1, cv.CV_64FC2);
  corners.data64F[0] = 0;
  corners.data64F[1] = 0;
  corners.data64F[2] = w;
  corners.data64F[3] = 0;
  corners.data64F[4] = w;
  corners.data64F[5] = h;
  corners.data64F[6] = 0;
  corners.data64F[7] = h;

  let warpedCorners = new cv.Mat();
  cv.perspectiveTransform(corners, warpedCorners, H);

  let xCoords = [];
  let yCoords = [];
  for (let i = 0; i < 4; i++) {
    xCoords.push(warpedCorners.data64F[i * 2]);
    yCoords.push(warpedCorners.data64F[i * 2 + 1]);
  }

  let minX = Math.min(...xCoords);
  let minY = Math.min(...yCoords);
  let maxX = Math.max(...xCoords);
  let maxY = Math.max(...yCoords);

  let tx = minX < 0 ? -minX : 0;
  let ty = minY < 0 ? -minY : 0;

  let T = cv.Mat.eye(3, 3, cv.CV_64FC1);
  T.data64F[2] = tx;
  T.data64F[5] = ty;

  let H_adj = T.mul(H, 1.0);

  let newWidth = Math.min(maxX - minX, MAX_WIDTH);
  let newHeight = Math.min(maxY - minY, MAX_HEIGHT);
  let newSize = new cv.Size(newWidth, newHeight);

  // Cleanup
  corners.delete();
  warpedCorners.delete();
  T.delete();

  return {H_adj, newSize};
}

function loadAndCorrectImage(img) {
  let src = cv.imread(img);
  cv.cvtColor(src, src, cv.COLOR_RGBA2BGR);

  let clusterLists = extractPotentialGridLines(src);

  if (clusterLists.length < 2) {
    return src;
  }

  let [clusterListH, clusterListV] = clusterLists;

  let linesH = linesToHomogeneous(clusterListH);
  let linesV = linesToHomogeneous(clusterListV);

  let vp1 = ransacVanishingPoint(linesH);
  let vp2 = ransacVanishingPoint(linesV);

  let H = estimateRectifyingHomography(vp1, vp2);

  let {H_adj, newSize} = adjustHomographyToView(src, H);
  let dst = new cv.Mat(newSize.height, newSize.width, cv.CV_8UC3);
  cv.warpPerspective(src, dst, H_adj, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, [0, 0, 0, 0]);

  // Cleanup
  H.delete();
  H_adj.delete();

  return dst;
}

function getSquaresFromMaskConnectedComponents(mask) {
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
    for (let j = i + 1; j < potentialSquares.length; j++) {
      const s2 = potentialSquares[j];
      // Compute intersection area
      const xA = Math.max(s1.x, s2.x);
      const yA = Math.max(s1.y, s2.y);
      const xB = Math.min(s1.x + s1.width, s2.x + s2.width);
      const yB = Math.min(s1.y + s1.height, s2.y + s2.height);
      const inter_w = Math.max(0, xB - xA);
      const inter_h = Math.max(0, yB - yA);
      const inter_area = inter_w * inter_h;
      const area1 = s1.width * s1.height;
      const area2 = s2.width * s2.height;
      if (area1 > 0 && area2 > 0) {
        const overlap_ratio = inter_area / Math.min(area1, area2);
        if (overlap_ratio > 0.3) {
          if (area1 < area2) {
            keepSquares[i] = false;
          } else {
            keepSquares[j] = false;
          }
        }
      }
    }
  }

  const filteredSquares = potentialSquares.filter((_, i) => keepSquares[i]);

  // Cleanup
  labels.delete();
  stats.delete();
  centroids.delete();

  return filteredSquares;
}


function findSquaresThresholding(src) {
  let img = src.clone();

  // Split channels
  let channels = new cv.MatVector();
  cv.split(img, channels);
  let blue = channels.get(0);
  let green = channels.get(1);
  let red = channels.get(2);

  // Create mask for really black pixels
  let mask = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
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

  let filteredSquares = getSquaresFromMaskConnectedComponents(mask);

  // Cleanup
  blue.delete(); green.delete(); red.delete();
  blueMask.delete(); greenMask.delete(); redMask.delete();
  low.delete(); high.delete();
  mask.delete(); kernel.delete();
  channels.delete(); img.delete();

  return filteredSquares;
}

function findSquaresDominantLines(src) {
  let clusterLists = extractPotentialGridLines(src, 160);

  if (clusterLists.length < 2) {
    return [];
  }

  let [clusterListH, clusterListV] = clusterLists;

  let gridMask = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);

  for (const cluster of clusterListH) {
    let [x1, y1, x2, y2] = cluster.mean;
    cv.line(gridMask, { x: x1, y: y1 }, { x: x2, y: y2 }, [255, 255, 255, 255], 1, cv.LINE_8, 0);
  }

  for (const cluster of clusterListV) {
    let [x1, y1, x2, y2] = cluster.mean;
    cv.line(gridMask, { x: x1, y: y1 }, { x: x2, y: y2 }, [255, 255, 255, 255], 1, cv.LINE_8, 0);
  }

  // Turn the edge 2px of the mask into 255
  for (let i = 0; i < gridMask.rows; i++) {
    for (let j = 0; j < gridMask.cols; j++) {
      if (i < 2 || i > gridMask.rows - 2 || j < 2 || j > gridMask.cols - 2) {
        gridMask.data[i * gridMask.cols + j] = 255;
      }
    }
  }

  cv.bitwise_not(gridMask, gridMask);
  let kernel = cv.Mat.ones(3, 3, cv.CV_8UC1);
  cv.morphologyEx(gridMask, gridMask, cv.MORPH_ERODE, kernel);

  let squares = getSquaresFromMaskConnectedComponents(gridMask);

  // Cleanup
  gridMask.delete();

  return squares;
}

function assignMetadataToSquares(squares, metadataArray) {
  const assignment = new Map();
  const usedMetadata = new Set();

  // Calculate square midpoints
  const squareMidpoints = squares.map(square => ({
    x: square.x + square.width / 2,
    y: square.y + square.height / 2,
  }));

  // For each square, find the closest unassigned metadata entry
  for (let squareIdx = 0; squareIdx < squares.length; squareIdx++) {
    const squareMidpoint = squareMidpoints[squareIdx];
    let closestMetadataIdx = null;
    let closestDistance = Infinity;

    for (let metaIdx = 0; metaIdx < metadataArray.length; metaIdx++) {
      if (usedMetadata.has(metaIdx)) {
        continue; // This metadata entry is already assigned
      }

      const metadata = metadataArray[metaIdx];
      const dx = squareMidpoint.x - metadata.midpointX;
      const dy = squareMidpoint.y - metadata.midpointY;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < closestDistance) {
        closestDistance = distance;
        closestMetadataIdx = metaIdx;
      }
    }

    // Assign the closest metadata to this square
    if (closestMetadataIdx !== null) {
      assignment.set(squareIdx, metadataArray[closestMetadataIdx]);
      usedMetadata.add(closestMetadataIdx);
    }
  }

  return assignment;
}

async function findSquares(src, metadataArray = null) {
  let filteredSquares = findSquaresThresholding(src);

  if (filteredSquares.length === 0) {
    filteredSquares = findSquaresDominantLines(src);
  }

  // If metadata is provided, assign it to squares by closest midpoint
  // Use metadata for any squares that can be matched (at least 1 match is valid)
  let metadataAssignment = null;
  if (metadataArray !== null && metadataArray.length > 0 && filteredSquares.length > 0) {
    metadataAssignment = assignMetadataToSquares(filteredSquares, metadataArray);
    if (metadataAssignment.size === 0) {
      metadataAssignment = null; // No matches, don't use metadata
    }
  }

  const hasMetadata = metadataAssignment !== null && metadataAssignment.size > 0;

  let squares = [];
  let worker = null;

  // Create worker if any squares need tesseract (either no metadata or some squares don't have metadata)
  const needsTesseract = !hasMetadata || metadataAssignment.size < filteredSquares.length;
  if (needsTesseract) {
    worker = await createWorker('eng');
    await worker.setParameters({
      tessedit_pageseg_mode: PSM.SINGLE_LINE,
      tessedit_char_whitelist: '0123456789xKkMm .',
    });
  }

  for (let i = 0; i < filteredSquares.length; i++) {
    const square = filteredSquares[i];
    let extraInfo = {};

    if (hasMetadata && metadataAssignment.has(i)) {
      // Use metadata if available for this square
      const metadata = metadataAssignment.get(i);
      if (metadata.quantity !== undefined) {
        extraInfo.quantity = metadata.quantity;
      }
      if (metadata.price !== undefined) {
        extraInfo.price = metadata.price;
      }
    } else if (square.y + 5 * square.height / 4 <= src.rows) {
      // Fall back to tesseract for squares without metadata
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
    }

    squares.push({...square, extraInfo});
  }

  if (worker) {
    await worker.terminate();
  }

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
  return new Promise(async (resolve) => {
    // Try to read metadata from the file
    let metadataArray = null;
    try {
      const arrayBuffer = await file.arrayBuffer();
      const metadataString = readPNGMetadata(arrayBuffer);

      if (metadataString) {
        // Parse metadata: midpoint_x::midpoint_y::quantity::price_string::...
        const parts = metadataString.split('::');
        if (parts.length % 4 === 0) {
          metadataArray = [];
          for (let i = 0; i < parts.length; i += 4) {
            const midpointX = parseFloat(parts[i]);
            const midpointY = parseFloat(parts[i + 1]);
            const quantity = parseInt(parts[i + 2]);
            const priceString = parts[i + 3];

            if (!isNaN(midpointX) && !isNaN(midpointY) && !isNaN(quantity) && priceString) {
              metadataArray.push({
                midpointX: midpointX,
                midpointY: midpointY,
                quantity: quantity,
                price: fromPriceString(priceString) || fromPriceString('30k'),
              });
            } else {
              // Invalid metadata entry, fall back to tesseract
              metadataArray = null;
              break;
            }
          }
        }
      }
    } catch (e) {
      // Silently fall back to tesseract if metadata reading fails
      metadataArray = null;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
      const src = loadAndCorrectImage(img);
      const squares = await findSquares(src, metadataArray);

      let tensors = [];
      let extraInfoArray = [];

      for (const square of squares) {
        const floatData = preprocessForONNX(src, square);
        tensors.push(floatData);
        extraInfoArray.push(square.extraInfo);
      }

      src.delete();

      resolve({
        dataArray: tensors,
        dims: [1, 3, 224, 224], // ONNX expects [N, C, H, W]
        extraInfoArray,
      });
    };
  });
}
