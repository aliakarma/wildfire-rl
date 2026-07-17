import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";

import type { RegionChoice } from "./data";

/**
 * Region choice persisted in the URL query (`?region=saudi`) so any exact view
 * can be shared or cited (§5.3). Persists across pages via the shared search params.
 */
export function useRegionChoice(): [RegionChoice, (r: RegionChoice) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get("region");
  const region: RegionChoice =
    raw === "california" || raw === "both" || raw === "saudi" ? raw : "saudi";

  const setRegion = useCallback(
    (r: RegionChoice) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("region", r);
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  return [region, setRegion];
}
