import React from 'react';
import {Figure, useStepper} from './Figure';
import {css} from './scene';

// Five Gaussians over three tiles, small enough to follow each one.
const GAUSSIANS = [
  {name: 'C', depth: 3.1, tiles: [1], color: [86, 180, 124], cx: 52},
  {name: 'E', depth: 4.2, tiles: [3], color: [176, 112, 224], cx: 312},
  {name: 'A', depth: 2.0, tiles: [1, 2], color: [235, 94, 72], cx: 116},
  {name: 'D', depth: 3.6, tiles: [2, 3], color: [76, 142, 232], cx: 244},
  {name: 'B', depth: 2.6, tiles: [2], color: [244, 179, 61], cx: 178},
];

const BY_DEPTH = [...GAUSSIANS].sort((a, b) => a.depth - b.depth);
const ENTRIES = BY_DEPTH.flatMap((g) => g.tiles.map((tile) => ({key: `${g.name}${tile}`, g, tile})));
const TILES = [1, 2, 3];
const BY_TILE = TILES.map((tile) => ENTRIES.filter((e) => e.tile === tile));

const CAPTIONS = [
  'The goal: a list for each tile, nearest Gaussian first. We start with the visible Gaussians in no particular order.',
  'Sort 1: sort the Gaussians by depth, nearest on the left. Each Gaussian is sorted once, however many tiles it touches.',
  'Copy each Gaussian once per tile it touches, keeping the order. A touches tiles 1 and 2, so it appears twice. The tile number is now on each copy.',
  'Sort 2: sort the copies by tile only. Copies with the same tile keep their order, so each tile’s list is already nearest first. That is the goal.',
];

const CELL = 52;
const X0 = 0;
const TILE_LABEL = 56;
const Y0 = 128;
const ROW = 42;

function layout(stage) {
  const pos = new Map();
  if (stage < 2) {
    const order = stage === 0 ? GAUSSIANS : BY_DEPTH;
    ENTRIES.forEach((e) => pos.set(e.key, {x: X0 + order.indexOf(e.g) * CELL, y: Y0, show: e.g.tiles[0] === e.tile}));
  } else if (stage === 2) {
    ENTRIES.forEach((e, i) => pos.set(e.key, {x: X0 + i * CELL, y: Y0, show: true}));
  } else {
    BY_TILE.forEach((list, row) => list.forEach((e, col) => pos.set(e.key, {x: TILE_LABEL + col * CELL, y: Y0 + row * ROW, show: true})));
  }
  return pos;
}

export default function Sorts() {
  const stepper = useStepper(4, 3400);
  const stage = stepper.step;
  const pos = layout(stage);
  const heading = ['Visible Gaussians', 'Sorted by depth (nearest first)', 'One copy per tile, same order', 'Sorted by tile: each row is one tile’s list'][stage];
  return (
    <Figure title="Getting each tile’s Gaussians in depth order" stepper={stepper} count={4} caption={CAPTIONS[stage]}>
      <svg viewBox="0 0 370 266" style={{maxWidth: 560}} role="img" aria-label="Five Gaussians sorted by depth, copied per tile, then sorted by tile">
        <text x="0" y="14" fontSize="12" fill="currentColor" fillOpacity="0.7">
          Three tiles and the Gaussians on them
        </text>
        {TILES.map((t, i) => (
          <g key={t}>
            <rect x={8 + i * 118} y="24" width="118" height="62" fill="none" stroke="currentColor" strokeOpacity="0.35" />
            <text x={14 + i * 118} y="82" fontSize="11" fill="currentColor" fillOpacity="0.65">
              tile {t}
            </text>
          </g>
        ))}
        {GAUSSIANS.map((g) => (
          <g key={g.name}>
            <ellipse cx={g.cx} cy="52" rx={g.tiles.length > 1 ? 48 : 22} ry="18" fill={css(g.color, 0.55)} />
            <text x={g.cx} y="56" textAnchor="middle" fontSize="12" fontWeight="600" fill="currentColor">
              {g.name}
            </text>
          </g>
        ))}
        <text x="0" y="110" fontSize="13" fontWeight="600" fill="currentColor">
          {heading}
        </text>
        {stage === 3 &&
          TILES.map((t, row) => (
            <text key={t} x="0" y={Y0 + row * ROW + 22} fontSize="12" fill="currentColor">
              tile {t}
            </text>
          ))}
        {ENTRIES.map((e) => {
          const p = pos.get(e.key);
          return (
            <g key={e.key} style={{transform: `translate(${p.x}px, ${p.y}px)`, transition: 'transform 800ms ease, opacity 400ms ease', opacity: p.show ? 1 : 0}}>
              <rect width={CELL - 8} height={ROW - 6} rx="5" fill={css(e.g.color)} />
              <text x={(CELL - 8) / 2} y="16" textAnchor="middle" fontSize="13" fontWeight="600" fill="#10131a">
                {e.g.name}
              </text>
              <text x={(CELL - 8) / 2} y="30" textAnchor="middle" fontSize="10" fill="#10131a">
                {stage >= 2 ? `tile ${e.tile}` : `depth ${e.g.depth}`}
              </text>
            </g>
          );
        })}
      </svg>
    </Figure>
  );
}
