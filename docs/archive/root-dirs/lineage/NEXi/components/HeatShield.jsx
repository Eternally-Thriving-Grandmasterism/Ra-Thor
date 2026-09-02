import { useState } from 'react';

export default function HeatShield({ onReady }) {
  const = useState(true);
  if (!healed) return null;
  return <span onClick={onReady} style={{ cursor: 'pointer' }}>🛡️</span>;
}
