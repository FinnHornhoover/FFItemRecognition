import React, { useState, useEffect } from 'react';
import { getPriceString, fromPriceString } from '../utils/PriceConversion';

function mergeResultsToInventory(prevInventory, newResults) {
  const inventory = [...prevInventory];

  const latestUpdateTime = inventory.reduce((max, result) => Math.max(max, result.updateTime), 0);
  if (newResults[0].updateTime <= latestUpdateTime) {
    return inventory;
  }

  // Remove isNew flag from all existing items when new items are imported
  inventory.forEach(item => {
    item.isNew = false;
  });

  newResults.forEach(result => {
    const existing = inventory.find(item => item.label === result.label);
    const importedQuantity = result.extraInfo.quantity || 1;
    const defaultPrice = fromPriceString(result.itemInfo.Price || '30k');
    const importedPrice = getPriceString(result.extraInfo.price || defaultPrice);

    if (existing) {
      existing.quantity += importedQuantity;
      existing.price = importedPrice;
      existing.updateTime = result.updateTime;
      existing.isNew = true; // Mark as newly updated
    } else {
      inventory.push({
        ...result,
        quantity: importedQuantity,
        price: importedPrice,
        isNew: true, // Mark as newly added
      });
    }
  });

  return inventory;
}

const ItemInventory = ({ newResults, onInventoryChange }) => {
  const [inventory, setInventory] = useState([]);
  const [draggedIndex, setDraggedIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);

  useEffect(() => {
    if (newResults && newResults.length > 0) {
      setInventory(prev => {
        const updated = mergeResultsToInventory(prev, newResults);
        // Sort: new items first, then by name
        return updated.sort((a, b) => {
          if (a.isNew && !b.isNew) return -1;
          if (!a.isNew && b.isNew) return 1;
          return 0;
        });
      });
    }
  }, [newResults]);

  useEffect(() => {
    if (onInventoryChange) {
      onInventoryChange(inventory);
    }
  }, [inventory, onInventoryChange]);

  const handleQuantityChange = (index, value) => {
    setInventory(prev => {
      const updated = [...prev];
      updated[index].quantity = parseInt(value) || 1;
      updated[index].isNew = false; // Remove new indicator when user modifies
      return updated;
    });
  };

  const handlePriceChange = (index, value) => {
    setInventory(prev => {
      const updated = [...prev];
      updated[index].price = value;
      updated[index].isNew = false; // Remove new indicator when user modifies
      return updated;
    });
  };

  const handleRemoveItem = (index) => {
    setInventory(prev => {
      const updated = [...prev];
      updated.splice(index, 1);
      return updated;
    });
  };

  const handleDragStart = (e, index) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', e.target.outerHTML);
  };

  const handleDragOver = (e, index) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverIndex(index);
  };

  const handleDragLeave = (e) => {
    setDragOverIndex(null);
  };

  const handleDrop = (e, dropIndex) => {
    e.preventDefault();
    setDragOverIndex(null);

    if (draggedIndex === null || draggedIndex === dropIndex) {
      setDraggedIndex(null);
      return;
    }

    setInventory(prev => {
      const updated = [...prev];
      const draggedItem = updated[draggedIndex];
      updated.splice(draggedIndex, 1);
      updated.splice(dropIndex, 0, draggedItem);
      return updated;
    });

    setDraggedIndex(null);
  };

  const handleDragEnd = (e) => {
    // Reset dragging state when drag ends (including outside the inventory area)
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  return (
    <div className="inventory-container">
      {inventory.map((item, i) => (
        <div
          key={item.label}
          className={`inventory-item ${draggedIndex === i ? 'dragging' : ''} ${dragOverIndex === i ? 'drag-over' : ''}`}
          draggable={inventory.length > 1}
          onDragStart={inventory.length > 1 ? (e) => handleDragStart(e, i) : undefined}
          onDragOver={inventory.length > 1 ? (e) => handleDragOver(e, i) : undefined}
          onDragLeave={inventory.length > 1 ? handleDragLeave : undefined}
          onDrop={inventory.length > 1 ? (e) => handleDrop(e, i) : undefined}
          onDragEnd={inventory.length > 1 ? handleDragEnd : undefined}
        >
          {/* Drag Handle Button */}
          <button
            className={`drag-handle ${inventory.length <= 1 ? 'drag-handle-disabled' : ''}`}
            title={inventory.length <= 1 ? "Need at least 2 items to reorder" : "Drag to reorder"}
            disabled={inventory.length <= 1}
          >
            ⋮⋮
          </button>

          <button
            onClick={() => handleRemoveItem(i)}
            className="modal-close-btn"
            style={{ position: 'absolute', top: '4px', right: '4px', zIndex: 1 }}
            aria-label="Remove item"
          >
            ×
          </button>
          {item.isNew && (
            <div style={{
              position: 'absolute',
              top: '40px',
              right: '24px',
              zIndex: 1,
              color: '#646cff',
              fontSize: '16px',
              fontWeight: 'bold'
            }}>
              *
            </div>
          )}
          <div className="item-icon-container-large">
            <img src={`/icons/${item.itemInfo['Icon']}`} alt={item.label} width={64} height={64} />
          </div>
          <div className="inventory-item-name">{item.itemInfo['Name']}</div>
          <div>Level: {item.itemInfo['Level']}</div>
          <div>Rarity: {item.itemInfo['Rarity']}</div>
          <div className="inventory-item-details">
            <label>Qty: </label>
            <input
              type="number"
              min="1"
              value={item.quantity}
              onChange={e => handleQuantityChange(i, e.target.value)}
            />
          </div>
          <div className="inventory-item-price">
            <label>Price: </label>
            <input
              type="text"
              value={item.price}
              onChange={e => handlePriceChange(i, e.target.value)}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

export default ItemInventory;
