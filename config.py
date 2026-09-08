"""
config
======

Per-system calibration configuration for the estimation pipeline.

Each entry names the data file and physical constants that define one
gas–liquid system.  Adding a system = add a CSV under ``data/`` + one
entry here; no code changes anywhere else (see ``core.calibration``).

``c_sat`` (saturation concentration, the units bridge) is *pressure*-
specific, so it is chosen per run, not per system: pass ``pressure_MPa``
to ``make_calibration`` and it is looked up from the CSV's pressure
column, or pass ``c_sat`` explicitly for systems whose CSV has no
pressure metadata.
"""

from pathlib import Path

from core.calibration import Calibration, c_sat_for_pressure

DATA_DIR = Path(__file__).resolve().parent / 'data'

# ----------------------------------------------------------------------
# CO2 + Weyburn reservoir brine, 27 °C  (Yang et al. 2006, Fig 6,
# digitized — see the CSV header for provenance).  gamma0/gamma_inf are
# the real digitized endpoints, not the placeholder's 72/35.
# ----------------------------------------------------------------------
CO2_BRINE_YANG2006 = dict(
    csv_path=DATA_DIR / 'calibration_co2_brine.csv',
    gamma0=70.55,       # mN/m at c = 0
    gamma_inf=36.64,    # mN/m at the highest digitized concentration
)

DEFAULT_SYSTEM = CO2_BRINE_YANG2006


def make_calibration(system=None, pressure_MPa=None, c_sat=None):
    """Build a ``Calibration`` for one run.

    Exactly one of ``pressure_MPa`` (looked up in the CSV's pressure
    column) or ``c_sat`` (explicit, in the CSV's concentration units)
    must be given.
    """
    if system is None:
        system = DEFAULT_SYSTEM
    if (pressure_MPa is None) == (c_sat is None):
        raise ValueError("give exactly one of pressure_MPa or c_sat")
    if c_sat is None:
        c_sat = c_sat_for_pressure(system['csv_path'], pressure_MPa)
    return Calibration(system['csv_path'],
                       gamma0=system['gamma0'],
                       gamma_inf=system['gamma_inf'],
                       c_sat=c_sat)
