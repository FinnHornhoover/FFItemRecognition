/**
 * Utility functions for reading and writing PNG metadata (tEXt chunks)
 */

const PNG_SIGNATURE = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
const CHUNK_TYPE_tEXt = [0x74, 0x45, 0x58, 0x74]; // "tEXt"

/**
 * Reads a PNG tEXt chunk from a PNG file buffer
 * @param {ArrayBuffer} buffer - The PNG file buffer
 * @param {string} keyword - The keyword to search for (default: "Inventory")
 * @returns {string|null} - The text content or null if not found
 */
export function readPNGMetadata(buffer, keyword = 'Inventory') {
  try {
    const dataView = new DataView(buffer);
    const uint8Array = new Uint8Array(buffer);

    // Check PNG signature
    for (let i = 0; i < PNG_SIGNATURE.length; i++) {
      if (uint8Array[i] !== PNG_SIGNATURE[i]) {
        return null; // Not a valid PNG
      }
    }

    let offset = PNG_SIGNATURE.length;

    // Read chunks
    while (offset < buffer.byteLength) {
      // Read chunk length (4 bytes, big-endian)
      if (offset + 4 > buffer.byteLength) break;
      const chunkLength = dataView.getUint32(offset, false);
      offset += 4;

      // Read chunk type (4 bytes)
      if (offset + 4 > buffer.byteLength) break;
      const chunkType = [];
      for (let i = 0; i < 4; i++) {
        chunkType.push(uint8Array[offset + i]);
      }
      offset += 4;

      // Check if it's a tEXt chunk
      let isTextChunk = true;
      for (let i = 0; i < 4; i++) {
        if (chunkType[i] !== CHUNK_TYPE_tEXt[i]) {
          isTextChunk = false;
          break;
        }
      }

      if (isTextChunk && chunkLength > 0) {
        // Read chunk data
        if (offset + chunkLength > buffer.byteLength) break;
        const chunkData = uint8Array.slice(offset, offset + chunkLength);
        offset += chunkLength;

        // Skip CRC (4 bytes)
        offset += 4;

        // Parse tEXt chunk: keyword (null-terminated) + text
        let keywordEnd = 0;
        while (keywordEnd < chunkData.length && chunkData[keywordEnd] !== 0) {
          keywordEnd++;
        }

        const foundKeyword = String.fromCharCode.apply(null, chunkData.slice(0, keywordEnd));
        if (foundKeyword === keyword) {
          const textStart = keywordEnd + 1;
          const textBytes = chunkData.slice(textStart);
          return String.fromCharCode.apply(null, textBytes);
        }
      } else {
        // Skip chunk data and CRC
        offset += chunkLength + 4;
      }
    }

    return null;
  } catch (e) {
    console.error('Error reading PNG metadata:', e);
    return null;
  }
}

/**
 * Writes a PNG tEXt chunk to a PNG file buffer
 * @param {ArrayBuffer} buffer - The original PNG file buffer
 * @param {string} text - The text content to write
 * @param {string} keyword - The keyword to use (default: "Inventory")
 * @returns {Blob} - A new PNG blob with the metadata added
 */
export function writePNGMetadata(buffer, text, keyword = 'Inventory') {
  try {
    const uint8Array = new Uint8Array(buffer);
    const dataView = new DataView(buffer);

    // Check PNG signature
    for (let i = 0; i < PNG_SIGNATURE.length; i++) {
      if (uint8Array[i] !== PNG_SIGNATURE[i]) {
        throw new Error('Not a valid PNG file');
      }
    }

    // Find the position after IHDR chunk (first chunk after signature)
    let offset = PNG_SIGNATURE.length;

    // Skip IHDR chunk (length + type + data + CRC = 4 + 4 + 13 + 4 = 25)
    if (offset + 4 > buffer.byteLength) {
      throw new Error('Invalid PNG structure');
    }
    const ihdrLength = dataView.getUint32(offset, false);
    offset += 4 + 4 + ihdrLength + 4; // length + type + data + CRC

    // Prepare tEXt chunk data
    const keywordBytes = new TextEncoder().encode(keyword);
    const textBytes = new TextEncoder().encode(text);
    const chunkDataLength = keywordBytes.length + 1 + textBytes.length; // keyword + null + text

    // Calculate CRC for tEXt chunk (CRC is calculated on chunk type + chunk data only)
    const crcData = new Uint8Array(4 + chunkDataLength);
    crcData.set(CHUNK_TYPE_tEXt, 0);
    crcData.set(keywordBytes, 4);
    crcData[keywordBytes.length + 4] = 0; // null terminator
    crcData.set(textBytes, keywordBytes.length + 5);

    // Simple CRC-32 calculation
    const crc = calculateCRC32(crcData);

    // Build new buffer
    const beforeChunk = uint8Array.slice(0, offset);
    const afterChunk = uint8Array.slice(offset);

    const newBuffer = new ArrayBuffer(
      beforeChunk.length +
      4 + // chunk length
      4 + // chunk type
      chunkDataLength + // chunk data
      4 + // CRC
      afterChunk.length
    );

    const newUint8Array = new Uint8Array(newBuffer);
    let newOffset = 0;

    // Copy data before insertion point
    newUint8Array.set(beforeChunk, newOffset);
    newOffset += beforeChunk.length;

    // Write chunk length
    const newDataView = new DataView(newBuffer);
    newDataView.setUint32(newOffset, chunkDataLength, false);
    newOffset += 4;

    // Write chunk type
    newUint8Array.set(CHUNK_TYPE_tEXt, newOffset);
    newOffset += 4;

    // Write chunk data
    newUint8Array.set(keywordBytes, newOffset);
    newOffset += keywordBytes.length;
    newUint8Array[newOffset++] = 0; // null terminator
    newUint8Array.set(textBytes, newOffset);
    newOffset += textBytes.length;

    // Write CRC
    newDataView.setUint32(newOffset, crc, false);
    newOffset += 4;

    // Copy data after insertion point
    newUint8Array.set(afterChunk, newOffset);

    return new Blob([newBuffer], { type: 'image/png' });
  } catch (e) {
    console.error('Error writing PNG metadata:', e);
    throw e;
  }
}

/**
 * Simple CRC-32 calculation (IEEE 802.3 polynomial)
 */
function calculateCRC32(data) {
  let crc = 0xFFFFFFFF;
  const polynomial = 0xEDB88320;

  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let j = 0; j < 8; j++) {
      if (crc & 1) {
        crc = (crc >>> 1) ^ polynomial;
      } else {
        crc = crc >>> 1;
      }
    }
  }

  return (crc ^ 0xFFFFFFFF) >>> 0;
}

