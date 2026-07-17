/**
 * GIS basemap for the interactive replay (§11.4).
 *
 * The image and cell geometry are emitted by scripts/export_basemaps.py from the same
 * GeoRenderer that renders the Phase-5/6 GIFs, so the viewer registers the 32x32 grid onto
 * the terrain exactly as the GIFs do.
 */

import { useEffect, useState } from "react";

import { dataUrl, useData, type Region } from "../lib/data";

export interface BasemapRegion {
  image: string;
  width: number;
  height: number;
  grid: number;
  displayName: string;
  assetLabel: string;
  assetColor: string;
  /** Cell centres as fractions of the image: x per column, y per row. */
  cellX: number[];
  cellY: number[];
  /** Uniform half-cell size, as a fraction of image width / height. */
  halfW: number;
  halfH: number;
  bounds: { lat: [number, number]; lon: [number, number] };
  scaleBarFrac: number;
}

export interface BasemapIndex {
  style: string;
  tileName: string;
  attribution: string;
  regions: Record<Region, BasemapRegion>;
}

const images = new Map<string, Promise<HTMLImageElement>>();

function loadImage(src: string): Promise<HTMLImageElement> {
  let p = images.get(src);
  if (!p) {
    p = new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`basemap: ${src}`));
      img.src = src;
    });
    images.set(src, p);
  }
  return p;
}

export interface LoadedBasemap {
  meta: BasemapRegion;
  attribution: string;
  img: HTMLImageElement;
}

/**
 * Basemap for one region, or undefined until the image has decoded. The canvas renders the
 * simulation regardless — a missing basemap degrades to a flat terrain fill rather than
 * blocking the replay.
 */
export function useBasemap(region: Region): LoadedBasemap | undefined {
  const { data } = useData<BasemapIndex>("basemaps/index.json");
  const [img, setImg] = useState<HTMLImageElement>();
  const meta = data?.regions?.[region];
  const src = meta ? dataUrl(`data/basemaps/${meta.image}`) : undefined;

  useEffect(() => {
    if (!src) return;
    let live = true;
    setImg(undefined);
    loadImage(src)
      .then((i) => live && setImg(i))
      .catch(() => live && setImg(undefined));
    return () => {
      live = false;
    };
  }, [src]);

  if (!data || !meta || !img) return undefined;
  return { meta, attribution: data.attribution, img };
}
