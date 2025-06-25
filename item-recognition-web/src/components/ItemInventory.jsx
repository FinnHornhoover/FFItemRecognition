import React, { useState, useEffect } from 'react';
import { getPriceString, fromPriceString } from '../utils/PriceConversion';

function mergeResultsToInventory(prevInventory, newResults) {
  const inventory = [...prevInventory];

  const latestUpdateTime = inventory.reduce((max, result) => Math.max(max, result.updateTime), 0);
  if (newResults[0].updateTime <= latestUpdateTime) {
    return inventory;
  }

  newResults.forEach(result => {
    const existing = inventory.find(item => item.label === result.label);
    const importedQuantity = result.extraInfo.quantity || 1;
    const defaultPrice = fromPriceString(result.itemInfo.Price || '30k');
    const importedPrice = getPriceString(result.extraInfo.price || defaultPrice);

    if (existing) {
      existing.quantity += importedQuantity;
      existing.price = importedPrice;
      existing.updateTime = result.updateTime;
    } else {
      inventory.push({
        ...result,
        quantity: importedQuantity,
        price: importedPrice,
      });
    }
  });

  return inventory;
}

const ItemInventory = ({ newResults, onInventoryChange }) => {
  const [inventory, setInventory] = useState([]);

  useEffect(() => {
    if (newResults && newResults.length > 0) {
      setInventory(prev => mergeResultsToInventory(prev, newResults));
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
      return updated;
    });
  };

  const handlePriceChange = (index, value) => {
    setInventory(prev => {
      const updated = [...prev];
      updated[index].price = value;
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

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
      {inventory.map((item, i) => (
        <div key={item.label} style={{ position: 'relative', border: '1px solid #ccc', borderRadius: '8px', padding: '12px', width: '220px', boxSizing: 'border-box', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <button
            onClick={() => handleRemoveItem(i)}
            className="modal-close-btn"
            style={{ position: 'absolute', top: '4px', right: '4px', zIndex: 1 }}
            aria-label="Remove item"
          >
            ×
          </button>
          <div className="item-icon-container-large">
            <img src={`/icons/${item.itemInfo['Icon']}`} alt={item.label} width={64} height={64} />
          </div>
          <div style={{ fontWeight: 'bold', marginTop: '8px' }}>{item.itemInfo['Name']}</div>
          <div>Level: {item.itemInfo['Level']}</div>
          <div>Rarity: {item.itemInfo['Rarity']}</div>
          <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center' }}>
            <label style={{ width: '50px' }}>Qty: </label>
            <input
              type="number"
              min="1"
              value={item.quantity}
              onChange={e => handleQuantityChange(i, e.target.value)}
              style={{ width: '60px' }}
            />
          </div>
          <div style={{ marginTop: '4px', display: 'flex', alignItems: 'center' }}>
            <label style={{ width: '50px' }}>Price: </label>
            <input
              type="text"
              value={item.price}
              onChange={e => handlePriceChange(i, e.target.value)}
              style={{ width: '60px' }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

export default ItemInventory;
