// src/ui/visual-effects/ValenceParticleField.tsx – Valence-modulated particle field background v1
// Floating mercy orbs, cyan-emerald glow, density/speed tied to valence
// MIT License – Autonomicity Games Inc. 2026

import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { currentValence } from '@/core/valence-tracker';

const ValenceParticleField: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [valence, setValence] = useState(currentValence.get());

  useEffect(() => {
    const unsubscribe = currentValence.subscribe(v => setValence(v));
    return unsubscribe;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: { x: number; y: number; size: number; speed: number; hue: number }[] = [];

    const createParticles = () => {
      const count = Math.floor(50 + valence * 150); // 50–200 particles
      particles.length = 0;
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: Math.random() * 3 + 1,
          speed: Math.random() * 0.5 + 0.1 * valence,
          hue: 180 + valence * 60 // cyan → emerald
        });
      }
    };

    createParticles();

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, 60%, ${0.3 + valence * 0.4})`;
        ctx.shadowBlur = 15;
        ctx.shadowColor = `hsla(${p.hue}, 100%, 70%, 0.6)`;
        ctx.fill();

        p.y -= p.speed;
        if (p.y < 0) p.y = canvas.height;
      });

      requestAnimationFrame(animate);
    };

    animate();

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      createParticles();
    };

    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [valence]);

  return (
    <motion.canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      initial={{ opacity: 0 }}
      animate={{ opacity: 0.6 + valence * 0.4 }}
      transition={{ duration: 2 }}
    />
  );
};

export default ValenceParticleField;
