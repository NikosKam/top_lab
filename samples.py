"""
Sample loading and caching  (supporting code -- students do not need to read this).

Flow:  ROOT file --uproot--> raw columnar arrays --(cached as parquet, so this only
       happens once)--> build_events() --> a plain Python list of Event objects.

Each Event has:
    event.weight        float
    event.trigger        bool
    event.met_px/py/pt  float
    event.muons         list of Muon objects
    event.jets           list of Jet objects
    event.truth           dict of four-vectors (MC samples only, None for data)

A Muon/Jet  has .pt, .eta, .phi, .mass, and two four-vectors can simply 
be added ( jet1 + jet2 ) to get their combined four-momentum.
"""
import os
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

# Where the ROOT files are. Locally: ../files. On Google Colab the setup cell sets LAB_DATA_DIR.
DATA_DIR = Path(os.environ.get("LAB_DATA_DIR", "../files"))
CACHE_DIR = Path("cache")

# tag -> file. The ORDER is the stacking order in the plots (same as the old example.py).
SAMPLES = {
    "Data": "data.root",
    "TTbar": "ttbar.root",
    "Wjets": "wjets.root",
    "DY": "dy.root",
    "WW": "ww.root",
    "WZ": "wz.root",
    "ZZ": "zz.root",
    "QCD": "qcd.root",
    "single Top": "single_top.root",
}

TREE_NAME = "events"

# Branch names, taken from the old MyAnalysis.py
_JET = ["Px", "Py", "Pz", "E", "btag", "ID"]
_MUON = ["Px", "Py", "Pz", "E", "Charge", "Iso"]
_TRUTH_PREFIX = {          # name we use in Event.truth -> branch name prefix in the ROOT file
    "lepton": "MClepton",
    "neutrino": "MCneutrino",
    "had_b": "MChadronicBottom",
    "lep_b": "MCleptonicBottom",
    "had_q": "MChadronicWDecayQuark",
    "had_qbar": "MChadronicWDecayQuarkBar",
}
_TRUTH_MASS = {            # GeV; same fixed values the old code used (see the lab notes on why)
    "lepton": 0.0, "neutrino": 0.0, "had_b": 4.8, "lep_b": 4.8, "had_q": 0.0, "had_qbar": 0.0,
}


# ----------------------------------------------------------------------------------
# Physics-object classes. Deliberately plain: a four-momentum (px, py, pz, E) plus
# whatever extra tags that object needs. Written in plain Python (no `vector` objects)
# because the samples hold ~700k events: one heavyweight object per particle would
# need several GB of memory. __slots__ keeps each object small.
# ----------------------------------------------------------------------------------
import math


class FourVector:
    """A four-momentum. Has .pt/.eta/.phi/.mass and can be added to any other
    four-vector (Jet, Muon, truth particle, or a previous sum) with `+`:
    jet1 + jet2 gives a W candidate, W candidate + b-jet gives a top candidate."""
    __slots__ = ("px", "py", "pz", "E")

    def __init__(self, px, py, pz, E):
        self.px, self.py, self.pz, self.E = px, py, pz, E

    @classmethod
    def from_mass(cls, px, py, pz, mass):
        return cls(px, py, pz, math.sqrt(px * px + py * py + pz * pz + mass * mass))

    @property
    def pt(self):
        return math.hypot(self.px, self.py)

    @property
    def eta(self):
        pt = self.pt
        if pt == 0:
            return math.copysign(float("inf"), self.pz) if self.pz else 0.0
        return math.asinh(self.pz / pt)

    @property
    def phi(self):
        return math.atan2(self.py, self.px)

    @property
    def mass(self):
        m2 = self.E ** 2 - self.px ** 2 - self.py ** 2 - self.pz ** 2
        return math.sqrt(m2) if m2 > 0 else -math.sqrt(-m2)   # same sign convention as ROOT

    def __add__(self, other):
        return FourVector(self.px + other.px, self.py + other.py, self.pz + other.pz, self.E + other.E)


class Jet(FourVector):
    """A reconstructed jet."""
    __slots__ = ("btag", "jetid")

    def __init__(self, px, py, pz, E, btag, jetid):
        super().__init__(px, py, pz, E)
        self.btag = btag      # b-tag discriminator (higher = more likely to come from a b-quark)
        self.jetid = jetid    # jet-quality flag

    def is_btagged(self, threshold=1.74):
        """True if this jet's b-tag discriminator is above `threshold`."""
        return self.btag > threshold


class Muon(FourVector):
    """A reconstructed muon."""
    __slots__ = ("charge", "iso")

    def __init__(self, px, py, pz, E, charge, iso):
        super().__init__(px, py, pz, E)
        self.charge = charge
        self.iso = iso     # isolation energy around the muon

    def is_isolated(self, relcut=0.09):
        """True if isolation / pT is below `relcut` (the muon is not sitting inside a jet)."""
        pt = self.pt if self.pt != 0 else 1e-9
        return (self.iso / pt) < relcut


class Event:
    """Everything belonging to one collision event."""
    __slots__ = ("weight", "trigger", "met_px", "met_py", "met_pt", "muons", "jets", "truth")

    def __init__(self, weight, trigger, met_px, met_py, muons, jets, truth=None):
        self.weight = weight
        self.trigger = trigger
        self.met_px = met_px
        self.met_py = met_py
        self.met_pt = math.hypot(met_px, met_py)
        self.muons = muons      # list of Muon
        self.jets = jets        # list of Jet
        self.truth = truth      # dict of four-vectors + "pdgid", or None for data


# ----------------------------------------------------------------------------------
# Reading + caching (unchanged mechanics: read once with uproot, cache as parquet)
# ----------------------------------------------------------------------------------

def _wanted_branches():
    b = ["EventWeight", "triggerIsoMu24", "MET_px", "MET_py", "MCleptonPDGid"]
    b += [f"Jet_{x}" for x in _JET] + [f"Muon_{x}" for x in _MUON]
    for prefix in _TRUTH_PREFIX.values():
        b += [f"{prefix}_{c}" for c in ("px", "py", "pz")]
    return b


def read_raw(tag, refresh=False):
    """Read the needed branches of one sample; cache them as parquet next to the notebook.
    The cache is optional: without pyarrow, the ROOT file is simply re-read every time."""
    src = DATA_DIR / SAMPLES[tag]
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run the setup cell at the top of the notebook "
                                f"(on Colab), or put the ROOT files in {DATA_DIR.resolve()}.")
    cache = CACHE_DIR / f"{src.stem}.parquet"
    if cache.exists() and not refresh and cache.stat().st_mtime > src.stat().st_mtime:
        try:
            return ak.from_parquet(cache)
        except ImportError:
            pass                      # no (recent enough) pyarrow: read the ROOT file instead

    with uproot.open(src) as f:
        tree = f[TREE_NAME]
        available = set(tree.keys())
        present = [b for b in _wanted_branches() if b in available]   # data has no truth branches
        raw = tree.arrays(present, library="ak")
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        ak.to_parquet(raw, cache)
    except ImportError as err:
        print(f"  (cache skipped: {err.args[0].splitlines()[0]} -- every load re-reads the ROOT file)")
    return raw


def _column(raw, name, default, n):
    """One flat branch as a plain Python list (or `default` repeated, if the branch is absent)."""
    return ak.to_numpy(raw[name]).tolist() if name in raw.fields else [default] * n


def _jagged(raw, prefix, names):
    """Jagged branches -> ({name: flat Python list of all values}, per-event counts).
    Converting whole columns at once is ~40x faster than indexing awkward arrays
    element by element inside a Python loop."""
    counts = ak.to_numpy(ak.num(raw[f"{prefix}_{names[0]}"])).tolist()
    flat = {x: ak.to_numpy(ak.flatten(raw[f"{prefix}_{x}"])).tolist() for x in names}
    return flat, counts


def build_events(raw):
    """Turn the raw branches into a plain Python list of Event objects, one per event.

    Runs once per sample each time the notebook starts (about a minute for all
    samples together); after this, the notebook only works with plain Python objects.
    """
    n = len(raw)
    weight = _column(raw, "EventWeight", 1.0, n)
    trigger = _column(raw, "triggerIsoMu24", False, n)
    met_px = _column(raw, "MET_px", 0.0, n)
    met_py = _column(raw, "MET_py", 0.0, n)

    mu, mu_n = _jagged(raw, "Muon", _MUON)
    jt, jt_n = _jagged(raw, "Jet", _JET)

    has_truth = "MCleptonPDGid" in raw.fields
    if has_truth:
        pdgid = _column(raw, "MCleptonPDGid", 0, n)
        truth_cols = {name: [_column(raw, f"{p}_{c}", 0.0, n) for c in ("px", "py", "pz")]
                      for name, p in _TRUTH_PREFIX.items()}

    events = []
    k_mu = k_jet = 0
    for i in range(n):
        muons = []
        for k in range(k_mu, k_mu + mu_n[i]):
            muons.append(Muon(mu["Px"][k], mu["Py"][k], mu["Pz"][k], mu["E"][k],
                              int(mu["Charge"][k]), mu["Iso"][k]))
        k_mu += mu_n[i]

        jets = []
        for k in range(k_jet, k_jet + jt_n[i]):
            jets.append(Jet(jt["Px"][k], jt["Py"][k], jt["Pz"][k], jt["E"][k],
                            jt["btag"][k], bool(jt["ID"][k])))
        k_jet += jt_n[i]

        truth = None
        if has_truth:
            truth = {"pdgid": int(pdgid[i])}
            for name, (px, py, pz) in truth_cols.items():
                truth[name] = FourVector.from_mass(px[i], py[i], pz[i], _TRUTH_MASS[name])

        events.append(Event(weight[i], bool(trigger[i]), met_px[i], met_py[i], muons, jets, truth))
    return events


def load_sample(tag, refresh=False):
    return build_events(read_raw(tag, refresh))


def load_all(refresh=False):
    """Dict tag -> list of Event. First call reads the ROOT files, later calls read the parquet cache."""
    events = {}
    for tag in SAMPLES:
        events[tag] = load_sample(tag, refresh)
        print(f"loaded {tag:<11s} {len(events[tag]):>9d} events")
    return events