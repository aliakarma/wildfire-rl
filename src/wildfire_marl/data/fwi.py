"""Canadian Fire Weather Index (FWI) System — standard equations.

Computes the daily FWI components (FFMC, DMC, DC, ISI, BUI, FWI) that Cell2Fire's
``Weather.csv`` carries, from screen-level weather (temperature °C, relative humidity %,
wind speed km/h, 24-h precipitation mm). This replaces made-up constants with the citable
standard used operationally across Canada:

* Van Wagner, C.E. (1987). *Development and structure of the Canadian Forest Fire Weather
  Index System*. Canadian Forestry Service, Forestry Technical Report 35.
* Van Wagner, C.E., Pickett, T.L. (1985). *Equations and FORTRAN program for the Canadian
  Forest Fire Weather Index System*. Canadian Forestry Service, Forestry Technical Report 33.

Implementation is the direct FORTRAN-report equation set (single-station, daily step).
Standard start-up values FFMC=85, DMC=6, DC=15 (Van Wagner 1987, §Startup).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Day-length factors (Van Wagner & Pickett 1985): DMC (Le) and DC (Lf), monthly, ~46°N.
# Used for both regions (36–41°N and 23–28°N ROIs); the induced error is small relative to
# the fuel-map sensitivity band and is ablated with it (Phase 10).
_DMC_DAY_LENGTH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
_DC_DAY_LENGTH = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


@dataclass
class FWIState:
    """Yesterday's moisture codes (standard start-up defaults)."""

    ffmc: float = 85.0
    dmc: float = 6.0
    dc: float = 15.0


def _ffmc(temp: float, rh: float, wind: float, rain: float, ffmc0: float) -> float:
    """Fine Fuel Moisture Code (Van Wagner & Pickett 1985, eqs. 1–10)."""
    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)
    if rain > 0.5:
        rf = rain - 0.5
        if mo > 150.0:
            mo = (
                mo
                + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
            )
        else:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        mo = min(mo, 250.0)
    ed = (
        0.942 * rh**0.679
        + 11.0 * math.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    )
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind) * (
            1.0 - (rh / 100.0) ** 8
        )
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (
            0.618 * rh**0.753
            + 10.0 * math.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
        )
        if mo < ew:
            kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind) * (
                1.0 - ((100.0 - rh) / 100.0) ** 8
            )
            kw = kl * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    return 59.5 * (250.0 - m) / (147.2 + m)


def _dmc(temp: float, rh: float, rain: float, dmc0: float, month: int) -> float:
    """Duff Moisture Code (eqs. 11–17)."""
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + math.exp(5.6348 - dmc0 / 43.43)
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)
        elif dmc0 <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc0)
        else:
            b = 6.2 * math.log(dmc0) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = max(0.0, 244.72 - 43.43 * math.log(mr - 20.0))
    else:
        pr = dmc0
    t = max(temp, -1.1)
    k = 1.894 * (t + 1.1) * (100.0 - rh) * _DMC_DAY_LENGTH[month - 1] * 1e-6
    return pr + 100.0 * k


def _dc(temp: float, rain: float, dc0: float, month: int) -> float:
    """Drought Code (eqs. 18–23)."""
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800.0 * math.exp(-dc0 / 400.0)
        qr = qo + 3.937 * rd
        dc_after_rain = max(0.0, 400.0 * math.log(800.0 / qr))
    else:
        dc_after_rain = dc0
    t = max(temp, -2.8)
    v = max(0.0, 0.36 * (t + 2.8) + _DC_DAY_LENGTH[month - 1])
    return dc_after_rain + 0.5 * v


def _isi(ffmc: float, wind: float) -> float:
    """Initial Spread Index (eqs. 24–26)."""
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 19.115 * math.exp(-0.1386 * m) * (1.0 + m**5.31 / 4.93e7)
    return 0.208 * math.exp(0.05039 * wind) * ff


def _bui(dmc: float, dc: float) -> float:
    """Buildup Index (eqs. 27–28)."""
    if dmc == 0.0 and dc == 0.0:
        return 0.0
    if dmc <= 0.4 * dc:
        return 0.8 * dmc * dc / (dmc + 0.4 * dc)
    return dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)


def _fwi(isi: float, bui: float) -> float:
    """Fire Weather Index (eqs. 28–30)."""
    if bui <= 80.0:
        bb = 0.1 * isi * (0.626 * bui**0.809 + 2.0)
    else:
        bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui)))
    if bb <= 1.0:
        return bb
    return math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)


def daily_fwi(
    temp: float,
    rh: float,
    wind: float,
    rain: float,
    month: int,
    prev: FWIState | None = None,
) -> tuple[FWIState, dict[str, float]]:
    """One daily FWI step. Returns (new state, {ffmc,dmc,dc,isi,bui,fwi})."""
    prev = prev or FWIState()
    rh = min(max(rh, 0.0), 100.0)
    ffmc = _ffmc(temp, rh, wind, rain, prev.ffmc)
    dmc = _dmc(temp, rh, rain, prev.dmc, month)
    dc = _dc(temp, rain, prev.dc, month)
    isi = _isi(ffmc, wind)
    bui = _bui(dmc, dc)
    fwi = _fwi(isi, bui)
    state = FWIState(ffmc=ffmc, dmc=dmc, dc=dc)
    return state, {
        "ffmc": round(ffmc, 2),
        "dmc": round(dmc, 2),
        "dc": round(dc, 2),
        "isi": round(isi, 2),
        "bui": round(bui, 2),
        "fwi": round(fwi, 2),
    }
