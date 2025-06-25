export function getPriceString(price) {
  if (price < 1000) {
    return `${price}`;
  } else if (price < 1000000) {
    const thousands = Math.round(price / 1000);
    const left = Math.round((price % 1000) / 100);

    if (left === 0) {
      return `${thousands}k`;
    } else {
      return `${thousands}.${left}k`;
    }
  } else {
    const millions = Math.round(price / 1000000);
    const left = Math.round((price % 1000000) / 10000);

    if (left === 0) {
      return `${millions}M`;
    } else {
      return `${millions}.${left}M`;
    }
  }
}

export function fromPriceString(priceStr) {
  const lower = priceStr.toLowerCase();
  if (lower.includes('k')) {
    return parseFloat(lower.split('k')[0]) * 1000;
  } else if (lower.includes('m')) {
    return parseFloat(lower.split('m')[0]) * 1000000;
  }
}
