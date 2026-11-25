"""Small helpers used by the notebook (histogram booking and filling)."""
import numpy as np
import hist

# Add a line here to book a new histogram for ALL samples:  name: (nbins, lo, hi, axis label)
RECO_HISTOGRAMS = {
    "n_muons":          (10, 0, 10, r"$N_{reco. muons}$"),
    "n_iso_mu":         (7, 0, 7, r"$N_{selected muons}$"),
    "n_jets":           (10, 0, 10, r"$N_{good jets}$"),
    "n_bjets":          (10, 0, 10, r"$N_{b-tagged jets}$"),
    "met":              (30, 5, 305, r"reco. $E_T^{miss}$ [GeV]"),
    "mu_pt":            (30, 0, 300, r"reco. muon $p_T$ [GeV] (after trigger only)"),
    "mu_pt_after_cuts": (30, 0, 300, r"reco. muon $p_T$ [GeV] (after all cuts)"),
    "mW":               (50, 0, 500, r"reconstructed $m_W$ [GeV]"),
    "mtop":             (20, 100, 300, r"reconstructed $m_{top}$ [GeV]"),
    "pt_top":           (30, 0, 350, r"reconstructed $p_{T,top}$ [GeV]"),
}


class Histogram(hist.Hist):
    """A hist.Hist that collects single values from fill_one() in a plain Python list and
    puts them into the histogram in one go when it is read (values(), plotting, ...).
    Filling one value at a time directly would be ~100x slower."""

    def _flush(self):
        buf = self.__dict__.get("_buf")
        if buf and buf[0]:
            xs, ws = buf
            super().fill(np.asarray(xs, dtype=float), weight=np.asarray(ws, dtype=float))
            xs.clear(); ws.clear()

    def values(self, *a, **k):
        self._flush(); return super().values(*a, **k)

    def variances(self, *a, **k):
        self._flush(); return super().variances(*a, **k)

    def view(self, *a, **k):
        self._flush(); return super().view(*a, **k)

    def sum(self, *a, **k):
        self._flush(); return super().sum(*a, **k)


def make_hist(nbins, lo, hi, label=""):
    """1D histogram that stores sum of weights and sum of weights^2 (= ROOT's Sumw2)."""
    return Histogram(hist.axis.Regular(nbins, lo, hi, label=label), storage=hist.storage.Weight())


def book_reco_histograms(*names):
    """Book the histograms with these names from RECO_HISTOGRAMS (all of them if no names are given)."""
    names = names or RECO_HISTOGRAMS.keys()
    return {name: make_hist(*RECO_HISTOGRAMS[name]) for name in names}


def fill_one(h, x, weight=1.0):
    """Fill ONE value x into histogram h, with an optional event weight (like ROOT's h->Fill(x, w))."""
    if "_buf" not in h.__dict__:
        h.__dict__["_buf"] = ([], [])
    xs, ws = h.__dict__["_buf"]
    xs.append(x)
    ws.append(weight)
    return h
