"""
Builds the student notebook for the lab:

    python build_notebooks.py

writes  lab_student.ipynb. (Solution-free copy: the instructor version is kept separately.)
"""
import json
import re

CELLS = []
def md(text):   CELLS.append(("markdown", text.strip("\n")))
def code(text): CELLS.append(("code", text.strip("\n")))

# --------------------------------------------------------------------------------------------
md(r"""
# $t\bar{t}$ production at the LHC: analysis lab

You will (1) study the *generated* (truth) particles in simulated $t\bar t$ events and (2) select
$t\bar t \to \mu + \text{jets}$ events in reconstructed data and simulation, reconstruct the W boson and the
top quark, and measure the $t\bar t$ cross section.

**How the data is organised.** Each sample (data, or one type of simulated background) is a plain Python
*list* of `Event` objects -- one `Event` per collision. Each event has:

* `event.muons` -- a **list** of muons seen in that event (could be 0, 1, 2, ...)
* `event.jets` -- a **list** of jets seen in that event
* `event.weight` -- how much this simulated event counts for (data always has weight 1)
* `event.truth` -- the *generated* particles (simulation only; `None` for real data)

Because every event can have a different number of muons/jets, you will go through the events **one at a
time with a `for` loop**, and go through the muons/jets of each event with a (possibly nested) `for` loop
too. For example:
```python
for event in tt:              # loop over every event in the sample
    for jet in event.jets:    # loop over every jet in this one event
        print(jet.pt)         # transverse momentum of this jet, in GeV
```
A jet or a muon has `.pt`, `.eta`, `.phi`, `.mass`, and two of
them can simply be **added** with `+` to get the four-momentum of the particle they came from, e.g.
`W_candidate = jet1 + jet2`.

The supporting code (reading the ROOT files, caching, plotting) lives in `samples.py`, `helpers.py` and
`plotting.py`. You do not need to read it, but you are welcome to.

**On Google Colab** (nothing to install): first click **File -> Save a copy in Drive** and work in
that copy -- otherwise your work is lost when you close the tab. Then run the setup cell below. Colab
forgets everything after about 90 minutes of inactivity; if that happens, just run all cells again.
""")

code(r"""
# Setup for Google Colab. When you run the notebook on your own computer, this cell does nothing.
import os, sys
if "google.colab" in sys.modules:
    import subprocess
    os.chdir("/content")
    if not os.path.exists("lab"):
        print("downloading the lab code and data (once per Colab session) ...")
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/NikosKam/top_lab.git", "lab"], check=True)
    os.chdir("/content/lab")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uproot", "awkward", "hist"], check=True)
    os.environ["LAB_DATA_DIR"] = "/content/lab/files"
    sys.path.insert(0, "/content/lab")
    print("Colab setup done.")
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from samples import load_all
from helpers import make_hist, fill_one, book_reco_histograms
from plotting import draw, plot_all

events = load_all()      # 1st run: reads the ROOT files. Later runs: reads a fast cache, so it's quick.
""")

code(r"""
# A small summary table: how many events, and how much do they add up to once weighted.
summary = {}
for tag, sample in events.items():
    total_weight = 0.0
    for event in sample:            # go through every event in this sample, one at a time
        total_weight += event.weight
    summary[tag] = {"events": len(sample), "sum of weights": total_weight}

pd.DataFrame(summary).T
""")

md(r"""
## Part 0: look at the data

`events["TTbar"]` is the list of simulated $t\bar t$ events. Let's look at the first one.
""")

code(r"""
tt = events["TTbar"]

first_event = tt[0]
print("number of muons in this event:", len(first_event.muons))
print("number of jets in this event: ", len(first_event.jets))

for jet in first_event.jets:
    print(f"  jet: pt = {jet.pt:6.1f} GeV   eta = {jet.eta:5.2f}   btag = {jet.btag:.2f}")
""")

md(r"""
**Questions**
1. How many jets does each of the first 5 events have? (Write a small `for` loop over `tt[:5]`.)
2. `event.met_pt` is the *missing transverse energy*. What does this quantity represent physically, and why
   is there no `met_pz`?
""")

# --------------------------------------------------------------------------------------------
md(r"""
## Part 1: generator-level ("truth") information, $t\bar t$ sample only

### Task 1: branching ratios of the W boson
Each simulated event has `event.truth["pdgid"]`: the PDG id of the generated lepton (particle and
antiparticle have the same |id|: $e$ = 11, $\mu$ = 13, $\tau$ = 15).
[PDG numbering scheme](https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf)

Go through every event, and count how many have a generated tau, a generated muon, and a generated
electron. The tau-counting is done for you -- add the `elif` branches for muon and electron.
""")

code(r"""
N_tau = 0
N_mu = 0     # <- you will count these
N_e = 0      # <- you will count these

for event in tt:                           # go through every simulated event, one at a time
    pdgid = abs(event.truth["pdgid"])       # |PDG id| of the generated lepton in this event

    if pdgid == 15:
        N_tau += 1
    # elif ...  # <- you will add code here to count muons and electrons

N = len(tt)
print(f"fraction of events with a generated tau:      {N_tau / N:.3f}")
# print(...)  # <- you will add code here to print the fractions for muons and electrons
""")

md(r"""
### Task 2: $p_T$ of the generated leptons
Fill one histogram per lepton flavour with the transverse momentum of the generated lepton
(`event.truth["lepton"].pt`). The tau histogram is filled for you -- add the muon and electron lines.
""")

code(r"""
h_tau_pt = make_hist(200, 0, 400, r"Generated lepton $p_T$ [GeV]")
h_mu_pt = make_hist(200, 0, 400, r"Generated lepton $p_T$ [GeV]")
h_e_pt = make_hist(200, 0, 400, r"Generated lepton $p_T$ [GeV]")

for event in tt:
    pdgid = abs(event.truth["pdgid"])
    lepton_pt = event.truth["lepton"].pt    # transverse momentum of the generated lepton

    if pdgid == 15:
        fill_one(h_tau_pt, lepton_pt)
    # elif ...  # <- you will add code here to fill the muon and electron histograms

fig, ax = plt.subplots()
draw(h_tau_pt, ax, label="tau")
# draw(...)   <- you will add code here to draw the muon and electron histograms
""")

md(r"""
### Task 3: W boson mass at truth level
Two four-vectors can be **added** (`a + b`) to get the four-vector of the particle they came from, and the
result has `.mass`, `.pt`, `.eta`, `.phi`. The truth four-vectors were built in `samples.py` from
$(p_x, p_y, p_z)$ and a fixed mass.

**Questions**: which units is the mass `4.8` (used for the b quarks) in? Why is the mass of the light
quarks set to `0`?

The hadronic W (from the two quarks) is reconstructed for you. Do the same for the **leptonic** W
(lepton + neutrino).
""")

code(r"""
h_mqq = make_hist(140, 10, 150, r"Generated $m_{qq}$ [GeV]")
h_mln = make_hist(140, 10, 150, r"Generated $m_{\ell\nu}$ [GeV]")

for event in tt:
    had_q = event.truth["had_q"]
    had_qbar = event.truth["had_qbar"]
    had_W = had_q + had_qbar               # adding two four-vectors reconstructs the W they came from
    fill_one(h_mqq, had_W.mass)

    # lepton = event.truth["lepton"]
    # neutrino = event.truth["neutrino"]
    # lep_W = ...          # add the lepton and neutrino four-vectors
    # fill_one(h_mln, lep_W.mass)

fig, ax = plt.subplots()
draw(h_mqq, ax, label="hadronic W")
# draw(...)  <- you will add code here to draw the leptonic W histogram
""")

md(r"""
### Task 4: top quark mass at truth level
Combine each W with its b quark (`event.truth["had_b"]`, `event.truth["lep_b"]`) to get the top quark, and
fill its mass and $p_T$. The hadronic side is done for you; do the same for the leptonic side.
""")

code(r"""
h_mhadtop = make_hist(30, 165, 180, r"Generated $m_{top}$ [GeV]")
h_pthadtop = make_hist(100, 0.1, 350, r"Generated $p_{T,top}$ [GeV]")
h_mleptop = make_hist(30, 165, 180, r"Generated $m_{top}$ [GeV]")
h_ptleptop = make_hist(100, 0.1, 350, r"Generated $p_{T,top}$ [GeV]")

for event in tt:
    had_W = event.truth["had_q"] + event.truth["had_qbar"]
    had_top = had_W + event.truth["had_b"]        # W boson + b-quark = top quark
    fill_one(h_mhadtop, had_top.mass)
    fill_one(h_pthadtop, had_top.pt)

    # lep_W = ...                       # same pattern as had_W, but with lepton + neutrino
    # lep_top = ...                     # lep_W + the leptonic b-quark
    # fill_one(h_mleptop, lep_top.mass)
    # fill_one(h_ptleptop, lep_top.pt)

# six plots: mass (hadronic alone / leptonic alone / both overlaid), then the same for pT
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

draw(h_mhadtop, axs[0, 0], label="hadronic top", color="tab:blue")
axs[0, 0].set_title("Hadronic top: mass")

draw(h_mleptop, axs[0, 1], label="leptonic top", color="tab:orange")
axs[0, 1].set_title("Leptonic top: mass")

draw(h_mhadtop, axs[0, 2], label="hadronic top", color="tab:blue")
draw(h_mleptop, axs[0, 2], label="leptonic top", color="tab:orange")
axs[0, 2].set_title("Mass: both")

draw(h_pthadtop, axs[1, 0], label="hadronic top", color="tab:blue")
axs[1, 0].set_title(r"Hadronic top: $p_T$")

draw(h_ptleptop, axs[1, 1], label="leptonic top", color="tab:orange")
axs[1, 1].set_title(r"Leptonic top: $p_T$")

draw(h_pthadtop, axs[1, 2], label="hadronic top", color="tab:blue")
draw(h_ptleptop, axs[1, 2], label="leptonic top", color="tab:orange")
axs[1, 2].set_title(r"$p_T$: both")

fig.tight_layout()
""")

# --------------------------------------------------------------------------------------------
md(r"""
## Part 2: reconstructed events, data vs. simulation

### Task 5: event selection
Selection criteria (defined once, used everywhere):
""")

code(r"""
MU_PT_CUT = 25.0        # GeV -- minimum transverse momentum for a "good" muon
MU_RELISO_CUT = 0.09    # isolation / pT -- below this, the muon is not sitting inside a jet
JET_PT_CUT = 30.0       # GeV
JET_ETA_CUT = 2.4
BTAG_CUT = 1.74         # b-tag discriminator threshold
M_W, M_TOP = 80.4, 172.5   # GeV -- known masses of the W boson and the top quark
LUMI = 50.0             # integrated luminosity -- CHECK THE UNIT (pb^-1?) before quoting a cross section
""")

md(r"""
#### Step 1: good muons and jets
`good_objects(event)` decides which muons and jets in one event are good enough to use, and splits the good
jets into b-tagged and non-b-tagged ("light") jets. Nothing to change here -- read it to see how each
requirement is applied.
""")

code(r"""
def good_objects(event):
    # good muons: pT above threshold AND isolated
    good_muons = []
    for mu in event.muons:
        if mu.pt > MU_PT_CUT and mu.is_isolated(MU_RELISO_CUT):
            good_muons.append(mu)

    # good jets: pT, |eta|, and jet-quality requirements
    good_jets = []
    for jet in event.jets:
        if jet.pt > JET_PT_CUT and abs(jet.eta) < JET_ETA_CUT and jet.jetid:
            good_jets.append(jet)

    # split the good jets into b-tagged and non-b-tagged ("light") jets
    bjets = []
    light_jets = []
    for jet in good_jets:
        if jet.is_btagged(BTAG_CUT):
            bjets.append(jet)
        else:
            light_jets.append(jet)

    return good_muons, good_jets, bjets, light_jets
""")

md(r"""
#### Step 2: distributions before the selection
Before choosing cuts, look at the samples. The only requirement here is the **trigger**: the data were only
recorded when the muon trigger fired, but the simulated $t\bar t$ sample also contains events where it did
not -- comparing them without the trigger would not be fair.

**Question**: in which of these distributions does $t\bar t$ (red) look most different from the backgrounds?
Those are the variables worth cutting on.
""")

code(r"""
def before_cuts(sample):
    h = book_reco_histograms("n_muons", "n_iso_mu", "n_jets", "n_bjets", "met", "mu_pt")
    for event in sample:
        if not event.trigger:
            continue
        good_muons, good_jets, bjets, light_jets = good_objects(event)
        weight = event.weight

        fill_one(h["n_muons"], len(event.muons), weight)
        fill_one(h["n_iso_mu"], len(good_muons), weight)
        fill_one(h["n_jets"], len(good_jets), weight)
        fill_one(h["n_bjets"], len(bjets), weight)
        fill_one(h["met"], event.met_pt, weight)
        if len(good_muons) >= 1:
            fill_one(h["mu_pt"], good_muons[0].pt, weight)
    return {"hists": h}


before = {}
for tag, sample in events.items():
    before[tag] = before_cuts(sample)
plot_all(before, pdf="before_cuts.pdf")
""")

md(r"""
#### Step 3: apply the cuts
`analyze(sample)` applies the cuts one at a time and counts how many (weighted) events survive each one.
For the events that survive every cut it reconstructs the W boson and the top quark with the function
`reconstruct_top()` that you will write in Task 6 (next cell).

**Your task in this cell**: the "4 good jets" and "2 b-jets" cuts are missing -- add them, following the
same pattern as the two cuts already there (`if ...: continue`, then add the event's weight to the
matching counter).
""")

code(r"""
def analyze(sample):
    h = book_reco_histograms("mu_pt_after_cuts", "mW", "mtop", "pt_top")

    w_initial = 0.0         # events surviving each cut, WEIGHTED (so backgrounds compare fairly)
    w_trigger = 0.0
    w_1muon = 0.0
    w_4jets = 0.0
    w_2bjets = 0.0

    for event in sample:                        # go through every event in this sample
        weight = event.weight
        w_initial += weight
        good_muons, good_jets, bjets, light_jets = good_objects(event)

        # --- the cuts, one at a time. `continue` skips the rest of the loop body,
        # so an event that fails a cut never reaches the ones after it.
        if not event.trigger:
            continue
        w_trigger += weight

        if len(good_muons) != 1:
            continue
        w_1muon += weight

        # TODO: require at least 4 good jets (if ...: continue), then add `weight` to w_4jets
        # TODO: require at least 2 b-tagged jets (if ...: continue), then add `weight` to w_2bjets

        fill_one(h["mu_pt_after_cuts"], good_muons[0].pt, weight)

        # an event needs >= 2 light jets (to build a W) and >= 1 b-jet (to build a top);
        # this only matters once the cuts above are filled in -- it protects against a crash
        if len(light_jets) < 2 or len(bjets) < 1:
            continue

        best_W, best_top = reconstruct_top(light_jets, bjets)     # Task 6, next cell
        fill_one(h["mW"], best_W.mass, weight)
        fill_one(h["mtop"], best_top.mass, weight)
        fill_one(h["pt_top"], best_top.pt, weight)

    cutflow = pd.Series({
        "initial": w_initial,
        "trigger": w_trigger,
        "1 muon": w_1muon,
        "4 jets": w_4jets,
        "2 b-jets": w_2bjets,
    })
    return {"hists": h, "cutflow": cutflow, "final_weighted": w_2bjets}
""")

md(r"""
### Task 6: reconstruct the W boson and the top quark
`reconstruct_top(light_jets, bjets)` gets the non-b-tagged jets and the b-tagged jets of **one** selected
event, and must return the best W boson candidate and the best top quark candidate:

* **W boson**: try every pair of light jets (two nested `for` loops: `i`, then `j > i`), add their
  four-vectors, and keep the pair whose mass is closest to `M_W`.
* **top quark**: add that W to each b-jet in turn, and keep the combination whose mass is closest to `M_TOP`.

The placeholder below just takes the first jets, so the rest of the notebook already runs. Replace it,
then run this cell **and** the next one again.
""")

code(r"""
def reconstruct_top(light_jets, bjets):
    # TODO (W boson): two nested for-loops over light_jets (i, then j > i), compute
    #   candidate = light_jets[i] + light_jets[j]
    # and keep whichever pair's mass is closest to M_W in `best_W`.
    #
    # TODO (top quark): one for-loop over bjets, compute
    #   candidate = best_W + bjet
    # and keep whichever is closest to M_TOP in `best_top`.
    #
    # placeholder so the rest of the notebook still runs before you fill this in:
    best_W = light_jets[0] + light_jets[1]
    best_top = best_W + bjets[0]
    return best_W, best_top
""")

code(r"""
results = {}
for tag, sample in events.items():
    print("analyzing", tag, "...")
    results[tag] = analyze(sample)

table = pd.DataFrame({tag: r["cutflow"] for tag, r in results.items()})
table.round(1)
""")

md(r"""
### Task 7: cross-section measurement
$\sigma = \dfrac{N_{data} - N_{bkg}}{\mathcal{L}\,A\,C}$ with the acceptance
$A = N^{fid}_{truth}/N^{gen}_{truth}$ and the efficiency $C = N^{reco}/N^{fid}_{truth}$.

The *fiducial region* (truth level): exactly one generated muon with $p_T>25$ GeV, and both b quarks with
$p_T>30$ GeV. Go through the $t\bar t$ sample once more and add up the weights of the fiducial events.
""")

code(r"""
fid_weight = 0.0
total_weight = 0.0

for event in tt:
    total_weight += event.weight
    pdgid = abs(event.truth["pdgid"])
    is_fiducial = (
        pdgid == 13
        and event.truth["lepton"].pt > 25
        and event.truth["had_b"].pt > 30
        and event.truth["lep_b"].pt > 30
    )
    if is_fiducial:
        fid_weight += event.weight

A = np.nan   # TODO: acceptance = (weighted fiducial events) / (weighted total events)
C = np.nan   # TODO: efficiency = (weighted, fully-selected ttbar events) / (weighted fiducial events)

final_cut = table.index[-1]
N_data = table.loc[final_cut, "Data"]
N_bkg = table.loc[final_cut].drop(["Data", "TTbar"]).sum()
sigma = (N_data - N_bkg) / (LUMI * A * C)
print(f"A = {A:.4f}   C = {C:.4f}   N_data = {N_data:.0f}   N_bkg = {N_bkg:.1f}")
print(f"sigma = {sigma:.1f}   (units follow from the luminosity unit)")
""")

md(r"""
### Plots: data vs. simulation after all cuts
Compare the muon $p_T$ after all cuts with the one after the trigger only (Step 2). **Which cut has the
largest impact on the background?** (The cutflow table above tells you cut by cut.)
""")

code(r"""
plot_all(results, pdf="after_cuts.pdf")
""")

md(r"""
### Extensions
* Add a new histogram: a line in `RECO_HISTOGRAMS` in `helpers.py`, its name in the `book_reco_histograms(...)`
  call of `before_cuts` or `analyze`, and a `fill_one(...)` call there. Compare signal vs. background shapes.
* Change the b-tag threshold or the jet $|\eta|$ cut. How do $N_{bkg}$ and $\sigma$ change? (Re-running
  takes only a few seconds.)
* Estimate the statistical uncertainty on $\sigma$.
""")

# --------------------------------------------------------------------------------------------
def to_nb():
    cells = []
    for kind, text in CELLS:
        lines = text.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


if __name__ == "__main__":
    with open("lab_student.ipynb", "w") as f:
        json.dump(to_nb(), f, indent=1)
    print("wrote lab_student.ipynb")
