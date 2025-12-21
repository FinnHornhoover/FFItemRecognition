import { useEffect, useState, useRef } from 'react';
import { loadModel, getEmbedding } from './utils/ImageEmbedder';
import { loadCrateEmbeddings, loadQuantizedEmbeddings, getNearestNeighbor, getNearestCrateNeighbor, getTopNNeighbors, areCrateEmbeddingsLoaded } from './utils/EmbeddingIndex';
import { preprocessImage } from './utils/ImageUtils';
import icon_labels from './labels/item_label_ids.json';
import truncated_item_info from './labels/item_info_truncated.json';
import crate_labels from './labels/crate_labels.json';
import './App.css'

// Component to display image with boxes drawn on detected items
function ImageWithBoxes({ imageFile, squareCoords, matches, boxType }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageFile || !squareCoords || !canvasRef.current) return;

    const img = new Image();
    const url = URL.createObjectURL(imageFile);

    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw boxes
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      squareCoords.forEach((coord, index) => {
        ctx.strokeRect(coord.x, coord.y, coord.width, coord.height);

        // Draw match number if this box is part of a match
        if (matches && matches.length > 0) {
          const match = matches.find(m =>
            boxType === 'crate' ? m.crateIndex === index : m.itemIndex === index
          );
          if (match) {
            ctx.fillStyle = '#00ff00';
            ctx.font = 'bold 16px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const textX = coord.x + coord.width / 2;
            const textY = coord.y + coord.height / 2;
            // Draw background circle for better visibility
            ctx.beginPath();
            ctx.arc(textX, textY, 12, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fill();
            // Draw number
            ctx.fillStyle = '#00ff00';
            ctx.fillText(match.matchNumber.toString(), textX, textY);
          }
        }
      });

      URL.revokeObjectURL(url);
    };

    img.src = url;

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [imageFile, squareCoords, matches, boxType]);

  if (!imageFile) return null;

  return (
    <div className="canvas-wrapper">
      <canvas
        ref={canvasRef}
        className="canvas"
      />
    </div>
  );
}

function App() {
  const [crateImage, setCrateImage] = useState(null);
  const [openedImage, setOpenedImage] = useState(null);
  const [crateImageFile, setCrateImageFile] = useState(null);
  const [openedImageFile, setOpenedImageFile] = useState(null);
  const [isProcessingCrates, setIsProcessingCrates] = useState(false);
  const [isProcessingItems, setIsProcessingItems] = useState(false);
  const [matches, setMatches] = useState([]); // Array of {id, crateType, itemLabel, itemInfo, midpointX, midpointY}
  const [editingMatchIds, setEditingMatchIds] = useState([]);
  const [editingCrateTypeMatchId, setEditingCrateTypeMatchId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadModel('/embedder.onnx');
    loadCrateEmbeddings('/crate_embeddings.bin');
    loadQuantizedEmbeddings(
      '/icon_embeddings.qvals.bin',
      '/icon_embeddings.starts.bin',
      '/icon_embeddings.steps.bin'
    );
  }, []);

  const processCrateImage = async (file) => {
    if (!file) return;

    // Check if crate embeddings are loaded
    if (!areCrateEmbeddingsLoaded()) {
      alert('Crate embeddings are still loading. Please wait a moment and try again.');
      return;
    }

    setIsProcessingCrates(true);
    // Clear all previous data when starting with a new crate image
    setOpenedImage(null);
    setOpenedImageFile(null);
    setMatches([]);
    try {
      const result = await preprocessImage(file);
      if (!result || !result.dataArray || result.dataArray.length === 0) {
        throw new Error('No items detected in the image');
      }
      const embeddings = await getEmbedding(result);
      if (!embeddings || embeddings.length === 0) {
        throw new Error('Failed to generate embeddings');
      }
      const crateResults = embeddings.map(getNearestCrateNeighbor);
      const crateTypes = crateResults.map(r => {
        if (r.index < 0 || r.index >= crate_labels.length) {
          throw new Error(`Invalid crate index: ${r.index}. Make sure crate_embeddings.bin matches crate_labels.json`);
        }
        return crate_labels[r.index];
      });

      // Store crate data with coordinates
      const crateData = embeddings.map((embedding, i) => ({
        embedding,
        crateType: crateTypes[i],
        midpointX: result.squareCoords[i].midpointX,
        midpointY: result.squareCoords[i].midpointY,
        distance: crateResults[i].distance,
        square: result.squareCoords[i]
      }));

      setCrateImage({ file, data: crateData, squareCoords: result.squareCoords });
    } catch (error) {
      console.error('Error processing crate image:', error);
      alert(`Error processing crate image: ${error.message || error}`);
    } finally {
      setIsProcessingCrates(false);
    }
  };

  const processOpenedImage = async (file) => {
    if (!file) return;
    setIsProcessingItems(true);
    try {
      const result = await preprocessImage(file);
      const embeddings = await getEmbedding(result);
      const itemResults = embeddings.map(getNearestNeighbor);
      const labels = itemResults.map(r => icon_labels[r.index][0]);
      const itemInfos = labels.map(l => truncated_item_info[l]);

      // Store item data with coordinates, including original square index
      const itemData = embeddings.map((embedding, i) => ({
        embedding,
        label: labels[i],
        itemInfo: itemInfos[i],
        midpointX: result.squareCoords[i].midpointX,
        midpointY: result.squareCoords[i].midpointY,
        distance: itemResults[i].distance,
        square: result.squareCoords[i],
        squareIndex: i // Store original index in squareCoords
      })).filter(item => item.label !== "00::0000"); // Filter out empty slots

      // Store all square coords for display
      setOpenedImage({ file, data: itemData, squareCoords: result.squareCoords });
    } catch (error) {
      console.error('Error processing opened image:', error);
      alert('Error processing opened image');
    } finally {
      setIsProcessingItems(false);
    }
  };

  useEffect(() => {
    if (!crateImage || !openedImage) return;

    const newMatches = [];
    let matchIdCounter = 0;
    const usedItemIndices = new Set(); // Track which items have been matched

    // For each crate, find the closest item by midpoint distance
    crateImage.data.forEach((crate, crateIndex) => {
      let closestItem = null;
      let closestItemIndex = -1;
      let minDistance = Infinity;

      openedImage.data.forEach((item, itemDataIndex) => {
        // Skip if this item was already matched
        if (usedItemIndices.has(itemDataIndex)) return;

        const dx = crate.midpointX - item.midpointX;
        const dy = crate.midpointY - item.midpointY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < minDistance) {
          minDistance = dist;
          closestItem = item;
          closestItemIndex = itemDataIndex;
        }
      });

      if (closestItem && closestItemIndex !== -1) {
        usedItemIndices.add(closestItemIndex);
        const matchNumber = matchIdCounter + 1; // 1-indexed for display
        newMatches.push({
          id: matchIdCounter++,
          crateType: crate.crateType,
          itemLabel: closestItem.label,
          itemInfo: closestItem.itemInfo,
          midpointX: crate.midpointX,
          midpointY: crate.midpointY,
          crateIndex: crateIndex,
          itemIndex: closestItem.squareIndex, // Use stored squareIndex
          matchNumber: matchNumber,
          disabled: false,
          embedding: closestItem.embedding // Store embedding for autofix
        });
      }
    });

    setMatches(newMatches);
  }, [crateImage, openedImage]);

  const handleCrateDragOver = (e) => {
    e.preventDefault();
  };

  const handleCrateDrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setCrateImageFile(files[0]);
      processCrateImage(files[0]);
    }
  };

  const handleOpenedDragOver = (e) => {
    e.preventDefault();
  };

  const handleOpenedDrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setOpenedImageFile(files[0]);
      processOpenedImage(files[0]);
    }
  };

  const handleCrateFileInput = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCrateImageFile(file);
      processCrateImage(file);
    }
  };

  const handleOpenedFileInput = (e) => {
    const file = e.target.files[0];
    if (file) {
      setOpenedImageFile(file);
      processOpenedImage(file);
    }
  };

  const handleToggleMatch = (id) => {
    setMatches(prev => prev.map(m =>
      m.id === id ? { ...m, disabled: !m.disabled } : m
    ));
  };

  const handleEditMatch = (matchId) => {
    setEditingMatchIds([matchId]);
    setSearchTerm('');
  };

  const handleSelectItem = (label, info) => {
    setMatches(prev => prev.map(m =>
      editingMatchIds.includes(m.id)
        ? { ...m, itemLabel: label, itemInfo: info }
        : m
    ));
    setEditingMatchIds([]);
    setSearchTerm('');
  };

  const handleEditCrateType = (matchId) => {
    setEditingCrateTypeMatchId(matchId);
  };

  const handleSelectCrateType = (crateType) => {
    setMatches(prev => prev.map(m =>
      m.id === editingCrateTypeMatchId
        ? { ...m, crateType: crateType }
        : m
    ));
    setEditingCrateTypeMatchId(null);
  };

  const handleAutofix = () => {
    if (!mostCommonLevel || Object.keys(levelCounts).length <= 1) {
      alert('No items to fix - all items are at the same level or no level data available.');
      return;
    }

    // Use the exact same logic as the red label display
    const redLabeledMatches = matches.filter(match => {
      const isRedLabeled = match.itemInfo.Level !== 0 &&
                          mostCommonLevel &&
                          Object.keys(levelCounts).length > 1 &&
                          String(match.itemInfo.Level) !== String(mostCommonLevel);
      return !match.disabled && isRedLabeled && match.embedding;
    });

    if (redLabeledMatches.length === 0) {
      alert('No items to fix - no red-labeled items found.');
      return;
    }

    let fixedCount = 0;
    const updatedMatches = matches.map(match => {
      // Use the exact same logic as the red label display
      const isRedLabeled = match.itemInfo.Level !== 0 &&
                          mostCommonLevel &&
                          Object.keys(levelCounts).length > 1 &&
                          String(match.itemInfo.Level) !== String(mostCommonLevel);
      const needsFix = !match.disabled && isRedLabeled && match.embedding;

      if (!needsFix) {
        return match; // Return unchanged
      }

      const topNeighbors = getTopNNeighbors(match.embedding, 10);

      // Find the first neighbor that matches the most common level
      for (const neighbor of topNeighbors) {
        const label = icon_labels[neighbor.index][0];
        const itemInfo = truncated_item_info[label];

        if (itemInfo && String(itemInfo.Level) === String(mostCommonLevel)) {
          // Found a match at the most common level
          fixedCount++;
          return {
            ...match,
            itemLabel: label,
            itemInfo: itemInfo
            // Keep the same embedding
          };
        }
      }

      // If no match found at most common level, leave as is
      return match;
    });

    setMatches(updatedMatches);
    alert(`Autofix completed. Fixed ${fixedCount} out of ${redLabeledMatches.length} item(s).`);
  };

  // Group matches by (crateType, itemLabel) and count, excluding disabled items
  const groupedMatches = matches
    .filter(match => !match.disabled)
    .reduce((acc, match) => {
      const key = `${match.crateType}::${match.itemLabel}`;
      if (!acc[key]) {
        acc[key] = {
          crateType: match.crateType,
          itemLabel: match.itemLabel,
          itemInfo: match.itemInfo,
          count: 0,
          matchIds: []
        };
      }
      acc[key].count++;
      acc[key].matchIds.push(match.id);
      return acc;
    }, {});

  const groupedMatchesArray = Object.values(groupedMatches);

  // Calculate most common item level (excluding disabled items)
  const levelCounts = matches
    .filter(match => !match.disabled)
    .reduce((acc, match) => {
      const level = String(match.itemInfo.Level); // Convert to string for consistency
      acc[level] = (acc[level] || 0) + 1;
      return acc;
    }, {});

  const mostCommonLevel = Object.keys(levelCounts).length > 0
    ? Object.keys(levelCounts).reduce((a, b) =>
        levelCounts[a] > levelCounts[b] ? a : b
      )
    : null;

  const handleCopyTSV = () => {
    navigator.clipboard.writeText(tsvText).then(() => {
      alert('Copied to clipboard!');
    }).catch(err => {
      console.error('Failed to copy:', err);
      // Fallback: select text
      const textarea = document.querySelector('textarea[readonly]');
      if (textarea) {
        textarea.select();
        document.execCommand('copy');
      }
    });
  };

  // Generate TSV text, sorted by crate type using crate_labels index
  const tsvText = groupedMatchesArray
    .sort((a, b) => {
      const indexA = crate_labels.indexOf(a.crateType);
      const indexB = crate_labels.indexOf(b.crateType);
      // If crate type not found in labels, put it at the end
      if (indexA === -1) return 1;
      if (indexB === -1) return -1;
      return indexA - indexB;
    })
    .map(g => `${g.crateType}\t${g.itemInfo.Name}\t${g.count}`)
    .join('\n');

  // Search results for item selection modal
  const searchResults = searchTerm.length > 0
    ? Object.entries(truncated_item_info)
        .map(([label, info]) => {
          const name = info.Name.toLowerCase();
          const term = searchTerm.toLowerCase();
          const index = name.indexOf(term);
          return index !== -1 ? { label, info, index, length: name.length } : null;
        })
        .filter(Boolean)
        .sort((a, b) => a.index - b.index || a.length - b.length)
        .slice(0, 5)
        .map(({ label, info }) => [label, info])
    : [];

  return (
    <div>
      <h1>Crate Drop Analyzer</h1>

      {/* Loading Spinners */}
      {isProcessingCrates && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <div className="loading-text">Processing crate image...</div>
          </div>
        </div>
      )}

      {isProcessingItems && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <div className="loading-text">Processing opened image...</div>
          </div>
        </div>
      )}

      {/* Two Drag and Drop Areas */}
      <div className="drag-drop-container">
        <div className="drag-drop-column">
          <h2>Crates Image</h2>
          <div
            className={`drag-drop-area ${crateImage ? 'has-image' : ''}`}
            onDragOver={handleCrateDragOver}
            onDrop={handleCrateDrop}
            onClick={() => document.getElementById('crate-file-input').click()}
          >
            <div className="drag-drop-content">
              <div className="drag-drop-icon">📦</div>
              <div className="drag-drop-text">
                <strong>Drop crate image here</strong>
                <br />
                or click to browse files
              </div>
              {crateImage && (
                <div className="detection-info">
                  {crateImage.data.length} crates detected
                </div>
              )}
            </div>
          </div>
          <input
            id="crate-file-input"
            type="file"
            accept="image/*"
            onChange={handleCrateFileInput}
            className="hidden-input"
          />
        </div>

        <div className="drag-drop-column">
          <h2>Opened Image</h2>
          <div
            className={`drag-drop-area ${openedImage ? 'has-image' : ''}`}
            onDragOver={handleOpenedDragOver}
            onDrop={handleOpenedDrop}
            onClick={() => document.getElementById('opened-file-input').click()}
          >
            <div className="drag-drop-content">
              <div className="drag-drop-icon">📦</div>
              <div className="drag-drop-text">
                <strong>Drop opened image here</strong>
                <br />
                or click to browse files
              </div>
              {openedImage && (
                <div className="detection-info">
                  {openedImage.data.length} items detected
                </div>
              )}
            </div>
          </div>
          <input
            id="opened-file-input"
            type="file"
            accept="image/*"
            onChange={handleOpenedFileInput}
            className="hidden-input"
          />
        </div>
      </div>

      {/* Image Display with Boxes */}
      {(crateImage || openedImage) && (
        <div className="image-display-section">
          <h2>Detected Items</h2>
          <div className="image-display-container">
            {crateImage && (
              <div className="image-display-item">
                <h3>Crates Image ({crateImage.data.length} crates)</h3>
                <ImageWithBoxes
                  imageFile={crateImage.file}
                  squareCoords={crateImage.squareCoords}
                  matches={matches}
                  boxType="crate"
                />
              </div>
            )}
            {openedImage && (
              <div className="image-display-item">
                <h3>Opened Image ({openedImage.squareCoords.length} boxes, {openedImage.data.length} items)</h3>
                <ImageWithBoxes
                  imageFile={openedImage.file}
                  squareCoords={openedImage.squareCoords}
                  matches={matches}
                  boxType="item"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Matches Display */}
      {matches.length > 0 && (
        <div className="results-section">
          <div className="tsv-header">
            <h2>Results</h2>
            <button
              onClick={handleAutofix}
              className="copy-button"
            >
              Autofix
            </button>
          </div>
          <div className="inventory-container">
            {matches.map((match) => {
              const levelClass = (match.itemInfo.Level !== 0 && mostCommonLevel && Object.keys(levelCounts).length > 1 && String(match.itemInfo.Level) !== String(mostCommonLevel))
                ? 'level-uncommon'
                : 'level-normal';

              const crateTypeClass = match.crateType === 'Standard' ? 'crate-type-standard' :
                                     match.crateType === 'Special' ? 'crate-type-special' :
                                     match.crateType === 'Sooper' ? 'crate-type-sooper' :
                                     match.crateType === 'Sooper Dooper' ? 'crate-type-sooper-dooper' :
                                     'crate-type-default';

              return (
                <div
                  key={match.id}
                  className={`inventory-item ${match.disabled ? 'inventory-item-disabled' : ''}`}
                >
                  <button
                    onClick={() => handleToggleMatch(match.id)}
                    className={`modal-close-btn toggle-button ${match.disabled ? 'toggle-button-disabled' : 'toggle-button-enabled'}`}
                    aria-label={match.disabled ? "Enable" : "Disable"}
                  >
                    {match.disabled ? '✓' : '×'}
                  </button>
                  <div className="item-icon-container-large">
                    <img
                      src={`/icons/${match.itemInfo.Icon}`}
                      alt={match.itemInfo.Name}
                      width={64}
                      height={64}
                    />
                  </div>
                  <div className="inventory-item-name">{match.itemInfo.Name}</div>
                  <div className={levelClass}>
                    Level: {match.itemInfo.Level}
                  </div>
                  <div className={crateTypeClass}>
                    Crate Type: {match.crateType}
                  </div>
                  <div className="edit-buttons-container">
                    <button
                      onClick={() => handleEditMatch(match.id)}
                      disabled={match.disabled}
                      className="edit-button"
                    >
                      Edit Item
                    </button>
                    <button
                      onClick={() => handleEditCrateType(match.id)}
                      disabled={match.disabled}
                      className="edit-button"
                    >
                      Edit Crate Type
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TSV Textbox */}
      {tsvText && (
        <div className="tsv-section">
          <div className="tsv-header">
            <h2>Export Data (Tab-Separated Values)</h2>
            <button
              onClick={handleCopyTSV}
              className="copy-button"
            >
              Copy to Clipboard
            </button>
          </div>
          <div className="tsv-stats">
            Total lines: {groupedMatchesArray.length} | Item total: {matches.filter(m => !m.disabled).length}
          </div>
          <textarea
            value={tsvText}
            readOnly
            className="tsv-textarea"
            onClick={(e) => e.target.select()}
          />
        </div>
      )}

      {/* Edit Item Modal */}
      {editingMatchIds.length > 0 && (
        <div className="modal-overlay">
          <div className="modal-container">
            <button
              onClick={() => {
                setEditingMatchIds([]);
                setSearchTerm('');
              }}
              className="modal-close-btn"
              aria-label="Close"
            >
              ×
            </button>
            <h2 className="modal-title">Select Item</h2>
            <input
              type="text"
              placeholder="Search item name..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="search-input"
              autoFocus
            />
            <div>
              {searchResults.length === 0 && searchTerm && (
                <div className="no-results">No matches found.</div>
              )}
              {searchResults.map(([label, info]) => (
                <div
                  key={label}
                  className="search-result-item"
                  onClick={() => handleSelectItem(label, info)}
                >
                  <div className="item-icon-container">
                    <img src={`/icons/${info.Icon}`} alt={label} width={40} height={40} />
                  </div>
                  <span className="item-name">{info.Name}</span>
                  <span className="item-details">Lv {info.Level} {info.Rarity}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Edit Crate Type Modal */}
      {editingCrateTypeMatchId !== null && (
        <div className="modal-overlay">
          <div className="modal-container">
            <button
              onClick={() => setEditingCrateTypeMatchId(null)}
              className="modal-close-btn"
              aria-label="Close"
            >
              ×
            </button>
            <h2 className="modal-title">Select Crate Type</h2>
            <div>
              {crate_labels.map((crateType) => (
                <div
                  key={crateType}
                  className="search-result-item"
                  onClick={() => handleSelectCrateType(crateType)}
                >
                  <span className={`item-name ${
                    crateType === 'Standard' ? 'crate-type-standard' :
                    crateType === 'Special' ? 'crate-type-special' :
                    crateType === 'Sooper' ? 'crate-type-sooper' :
                    crateType === 'Sooper Dooper' ? 'crate-type-sooper-dooper' :
                    'crate-type-default'
                  }`}>
                    {crateType}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App
