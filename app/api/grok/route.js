// app/api/grok/route.js
// Rathor-NEXi Grok Proxy on Vercel – Key-hidden + CORS + Streaming
// MIT License – Autonomicity Games Inc. 2026

import { NextResponse } from 'next/server';

export const runtime = 'edge'; // Fast global edge

export async function OPTIONS() {
  return new NextResponse(null, {
    headers: {
      'Access-Control-Allow-Origin': 'https://rathor.ai',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

export async function POST(request) {
  try {
    const body = await request.json();

    const xaiResponse = await fetch('https://api.x.ai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.XAI_API_KEY}`, // Set in Vercel Env Vars
      },
      body: JSON.stringify(body),
    });

    if (!xaiResponse.ok) {
      const error = await xaiResponse.text();
      return new NextResponse(`xAI error: ${xaiResponse.status} - ${error}`, {
        status: xaiResponse.status,
        headers: { 'Access-Control-Allow-Origin': 'https://rathor.ai' },
      });
    }

    // Stream directly with CORS
    return new NextResponse(xaiResponse.body, {
      status: xaiResponse.status,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': 'https://rathor.ai',
      },
    });

  } catch (err) {
    return new NextResponse('Proxy lattice error: ' + err.message, {
      status: 500,
      headers: { 'Access-Control-Allow-Origin': 'https://rathor.ai' },
    });
  }
}
