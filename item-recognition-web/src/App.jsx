import { useEffect, useState } from 'react';
import { loadModel, getEmbedding } from './utils/ImageEmbedder';
import { loadQuantizedEmbeddings, getNearestNeighbor } from './utils/EmbeddingIndex';
import { preprocessImage } from './utils/ImageUtils';
import ItemInventory from './components/ItemInventory';
import icon_labels from './labels/item_label_ids.json';
import truncated_item_info from './labels/item_info_truncated.json';
import './App.css'

function App() {
  const [_, setResults] = useState([]);
  const [newResults, setNewResults] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [inventoryKey, setInventoryKey] = useState(0); // for resetting inventory
  const [currentInventory, setCurrentInventory] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);

  useEffect(() => {
    loadModel('/embedder.onnx');
    loadQuantizedEmbeddings(
      '/icon_embeddings.qvals.bin',
      '/icon_embeddings.starts.bin',
      '/icon_embeddings.steps.bin'
    );
  }, []);

  const handleEmbedReady = async (tensors) => {
    const extraInfoArray = tensors.extraInfoArray;
    const embeddings = await getEmbedding(tensors);
    const results = embeddings.map(getNearestNeighbor);
    const labels = results.map(r => icon_labels[r.index][0]);
    const item_infos = labels.map(l => truncated_item_info[l]);

    let output = [];
    for (let i = 0; i < embeddings.length; i++) {
      if (labels[i] === "00::0000") {
        continue;
      }
      output.push({
        label: labels[i],
        distance: results[i].distance,
        itemInfo: item_infos[i],
        extraInfo: extraInfoArray[i],
        updateTime: Date.now(),
      });
    }

    setResults(output);
    setNewResults(output); // Pass to inventory
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    const tensors = await preprocessImage(file);
    handleEmbedReady(tensors);

    // Clear the file input so the same file can be uploaded again
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragAreaClick = () => {
    document.getElementById('file-input').click();
  };

  // Add Item Modal logic
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

  const handleAddItem = (label, info) => {
    setNewResults([{ label, distance: 0, itemInfo: info, updateTime: Date.now(), extraInfo: { quantity: 1, price: 30000 } }]);
    setShowAddModal(false);
    setSearchTerm('');
  };

  const handleClearAll = () => {
    setInventoryKey(k => k + 1); // force remount ItemInventory
    setNewResults([]);
  };

  const handleExportImage = () => {
    // Use current inventory data
    const itemCount = currentInventory.length || 0;
    if (itemCount === 0) {
      alert('No items to export');
      return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Calculate grid dimensions to minimize area (prefer more rows, fewer columns)
    const cols = Math.ceil(Math.sqrt(itemCount));
    const rows = Math.ceil(itemCount / cols);

    // Item dimensions: 64px icon + 20px text height + 3px gap
    const itemWidth = 64;
    const itemHeight = 84; // 64 + 20 for text
    const gap = 10;

    // Canvas dimensions
    const canvasWidth = cols * itemWidth + (cols - 1) * gap;
    const canvasHeight = rows * itemHeight + (rows - 1) * gap;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Set black background for the entire canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Track loaded images
    let loadedImages = 0;
    const totalImages = itemCount;

    const triggerDownload = () => {
      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'inventory-export.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    };

    const drawItem = (item, x, y) => {
      // Draw background image for this item slot
      const itemBgImg = new Image();
      itemBgImg.crossOrigin = 'anonymous';

      const drawItemContent = () => {
        // Draw quantity and price text
        const quantityText = `${item.quantity || 1}x`;
        const priceText = item.price || '30k';

        ctx.fillText(quantityText, x + 2, y + 76);
        const priceTextWidth = ctx.measureText(priceText).width;
        ctx.fillText(priceText, x + 64 - priceTextWidth - 2, y + 76);

        loadedImages++;
        if (loadedImages === totalImages) {
          triggerDownload();
        }
      };

      itemBgImg.onload = () => {
        // Draw background image for this item slot
        ctx.drawImage(itemBgImg, x, y, 64, 64);

        // Now load and draw the item icon
        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = () => {
          ctx.drawImage(img, x, y, 64, 64);
          drawItemContent();
        };

        img.onerror = () => {
          // Fallback: draw a placeholder rectangle
          ctx.fillStyle = '#333333';
          ctx.fillRect(x, y, 64, 64);
          ctx.fillStyle = '#ffffff';
          drawItemContent();
        };

        img.src = `/icons/${item.itemInfo.Icon}`;
      };

      itemBgImg.onerror = () => {
        // If background fails, just draw the item directly
        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = () => {
          ctx.drawImage(img, x, y, 64, 64);
          drawItemContent();
        };

        img.onerror = () => {
          // Fallback: draw a placeholder rectangle
          ctx.fillStyle = '#333333';
          ctx.fillRect(x, y, 64, 64);
          ctx.fillStyle = '#ffffff';
          drawItemContent();
        };

        img.src = `/icons/${item.itemInfo.Icon}`;
      };

      itemBgImg.src = '/inv_empty.png';
    };

    const drawAllItems = () => {
      // Set white text
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px DejaVu Serif';
      ctx.textAlign = 'left';

      // Draw items
      let itemIndex = 0;
      for (let row = 0; row < rows && itemIndex < itemCount; row++) {
        for (let col = 0; col < cols && itemIndex < totalImages; col++) {
          const x = col * (itemWidth + gap);
          const y = row * (itemHeight + gap);

          const item = currentInventory[itemIndex];
          if (item) {
            drawItem(item, x, y);
          }
          itemIndex++;
        }
      }

      // Fallback: if no images load at all, still trigger download after a timeout
      setTimeout(() => {
        if (loadedImages === 0) {
          triggerDownload();
        }
      }, 2000);
    };

    // Start drawing items (each will load its own background)
    drawAllItems();
  };

  return (
    <>
    <div>
      <h1>Retrobution Shoplist</h1>

      {/* Drag and Drop Area */}
      <div
        className={`drag-drop-area ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleDragAreaClick}
      >
        <div className="drag-drop-content">
          <div className="drag-drop-icon">📁</div>
          <div className="drag-drop-text">
            <strong>Import inventory / bank screenshot here</strong>
            <br />
            or click to browse files (.png, .jpg, .jpeg, .webp)
          </div>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      <div className="button-group">
        <button onClick={() => setShowAddModal(true)}>Add Item</button>
        <button onClick={handleClearAll}>Clear All</button>
        <button onClick={handleExportImage}>Export Image</button>
      </div>

      {showAddModal && (
        <div className="modal-overlay">
          <div className="modal-container">
            <button
              onClick={() => setShowAddModal(false)}
              className="modal-close-btn"
              aria-label="Close"
            >
              ×
            </button>
            <h2 className="modal-title">Add Item</h2>
            <input
              type="text"
              placeholder="Search item name..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="search-input"
              autoFocus
            />
            <div>
              {searchResults.length === 0 && searchTerm && <div className="no-results">No matches found.</div>}
              {searchResults.map(([label, info]) => (
                <div
                  key={label}
                  className="search-result-item"
                  onClick={() => handleAddItem(label, info)}
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
      <ItemInventory key={inventoryKey} newResults={newResults} onInventoryChange={setCurrentInventory} />
    </div>
    </>
  );
}

export default App
