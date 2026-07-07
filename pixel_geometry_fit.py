#!/usr/bin/env python3
"""
Amplitude maps and pixel-to-pixel boundary crossings for the 9 "pixel"
interpolation cells built from groups of 4 pmax channels (2x2 sliding sum
over the 4x4 channel grid):

  pixel0: 14,15,8,9   pixel1: 15,13,10,8  pixel2: 13,16,11,10
  pixel3: 9,8,6,7     pixel4: 8,10,5,6    pixel5: 10,11,4,5
  pixel6: 7,6,2,3     pixel7: 6,5,1,2     pixel8: 5,4,12,1

Physical channel layout (confirmed):
  row0 (top, high y):    14  15  13  16
  row1:                   9   8  10  11
  row2:                   7   6   5   4
  row3 (bottom, low y):   3   2   1  12
  columns left->right (low x -> high x).

pixel(R,C) = 3*R + C, R=0 top row of pixels .. R=2 bottom row; C=0 left
column .. C=2 right column.

A single pixel's own amplitude map does NOT have sharp edges except where
it touches the true sensor boundary: the induced signal on a given group
of 4 channels decays gradually over much of the sensor, so an absolute
50%-of-peak threshold on one pixel alone gives broad, overlapping regions
(verified interactively). Instead, the boundary between two NEIGHBOURING
pixels (sharing 2 channels) is found from the DIFFERENCE of their sums
(sum_A - sum_B): this crosses zero right at the geometric boundary between
the two cells. The crossing is located by scanning the sliced 1D profile
of the difference for its most significant sign change (largest jump,
bins with too few entries discarded to reject noise), which is then
refined with an erf S-curve fit in a narrow window around it -- same 50%
criterion as the whole-sensor edges, just applied to the difference of two
neighbouring cells instead of an absolute amplitude.

NOTE: unlike the whole-sensor edges, these internal boundaries turned out
NOT to all line up into a clean rotated rectangular grid -- some pairs
(especially involving channel rows 2/3) give crossings that are hard to
reconcile with a single straight dividing line. This script therefore
reports the 12 raw pairwise crossings without forcing a grid model; each
is flagged with the profile's local contrast (jump size) as a rough
reliability indicator.

Usage:
  ./run_root_python.sh pixel_geometry_fit.py [root_file] [maps.png]
"""
import sys
import ROOT

DEFAULT_FILE = "/Volumes/DCRSD_2/TB11/stats_Run5_tracks.root"
NBINS_X, XLO, XHI = 170, -2.0, -0.3
NBINS_Y, YLO, YHI = 170, -2.6, -0.9

PIXEL_CHANNELS = {
    0: (14, 15, 8, 9), 1: (15, 13, 10, 8), 2: (13, 16, 11, 10),
    3: (9, 8, 6, 7), 4: (8, 10, 5, 6), 5: (10, 11, 4, 5),
    6: (7, 6, 2, 3), 7: (6, 5, 1, 2), 8: (5, 4, 12, 1),
}
GRID = {i: (i // 3, i % 3) for i in range(9)}  # idx -> (R, C); R=0 top, C=0 left

# Nominal centers of each pixel-row / pixel-column, used only to pick where
# to slice the 2D difference map when looking for a given boundary. Derived
# from the whole-sensor box (sensor_geometry_fit.py) assuming even spacing.
SENSOR_X = (-1.886, -0.408)
SENSOR_Y = (-2.506, -1.027)
ROW_Y_CENTER = {0: -1.520, 1: -2.013, 2: -2.383}  # R=0 top (high y) .. R=2 bottom
COL_X_CENTER = {0: -1.393, 1: -0.901, 2: -0.531}  # C=0 left (low x) .. C=2 right
SLICE_HALF = 0.20
FIT_MARGIN = 0.15
MIN_ENTRIES = 50


def z_expr(channels):
    return "+".join(f"pmax[{c}]" for c in channels)


def build_profile2d(tree, zexpr):
    draw_str = (
        f"({zexpr}):(40*(y_pos1-y_pos2)+y_pos1):(40*(x_pos1-x_pos2)+x_pos1)"
        f">>h({NBINS_X},{XLO},{XHI},{NBINS_Y},{YLO},{YHI})"
    )
    tree.Draw(draw_str, "", "prof goff")
    return ROOT.gDirectory.Get("h").Clone(f"hclone_{abs(hash(zexpr)) % 10_000_000}")


def rough_crossing(h1d, min_entries=MIN_ENTRIES):
    """Most significant sign change in the profile (largest jump), ignoring
    low-statistics bins so noise near baseline doesn't get picked up."""
    xs, ys = [], []
    for i in range(1, h1d.GetNbinsX() + 1):
        if h1d.GetBinEntries(i) >= min_entries:
            xs.append(h1d.GetXaxis().GetBinCenter(i))
            ys.append(h1d.GetBinContent(i))
    candidates = []
    for i in range(len(ys) - 1):
        if (ys[i] < 0) != (ys[i + 1] < 0):
            frac = -ys[i] / (ys[i + 1] - ys[i])
            x_cross = xs[i] + frac * (xs[i + 1] - xs[i])
            candidates.append((abs(ys[i + 1] - ys[i]), x_cross))
    if not candidates:
        return None, 0.0
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][0]


def refine_with_erf(h1d, crossing, margin=FIT_MARGIN):
    lo, hi = crossing - margin, crossing + margin
    v_lo = h1d.GetBinContent(h1d.GetXaxis().FindBin(lo))
    v_hi = h1d.GetBinContent(h1d.GetXaxis().FindBin(hi))
    f = ROOT.TF1(f"ferf_{abs(hash((lo, hi, v_lo))) % 10_000_000}",
                 "[0]+[1]*TMath::Erf((x-[2])/(TMath::Sqrt(2)*[3]))", lo, hi)
    f.SetParameters((v_lo + v_hi) / 2, (v_hi - v_lo) / 2, crossing, margin / 3)
    status = int(h1d.Fit(f, "RQS").Status())
    if status == 0 and lo < f.GetParameter(2) < hi:
        return f.GetParameter(2), f.GetParError(2), True
    return crossing, None, False


def boundary_crossing(tree, idx_a, idx_b, direction, slice_center, tag):
    """direction 'x': horizontal-neighbour pair, profile vs x at fixed y-slice.
    direction 'y': vertical-neighbour pair, profile vs y at fixed x-slice."""
    diff_expr = f"({z_expr(PIXEL_CHANNELS[idx_a])})-({z_expr(PIXEL_CHANNELS[idx_b])})"
    h = build_profile2d(tree, diff_expr)
    lo, hi = slice_center - SLICE_HALF, slice_center + SLICE_HALF
    if direction == "x":
        prof = h.ProfileX(f"p_{tag}", h.GetYaxis().FindBin(lo), h.GetYaxis().FindBin(hi))
    else:
        prof = h.ProfileY(f"p_{tag}", h.GetXaxis().FindBin(lo), h.GetXaxis().FindBin(hi))
    cross, jump = rough_crossing(prof)
    if cross is None:
        return None
    pos, err, converged = refine_with_erf(prof, cross)
    return {"pos": pos, "err": err, "converged": converged, "jump": jump}


def main():
    root_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    out_png = sys.argv[2] if len(sys.argv) > 2 else "pixel_maps.png"

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        sys.exit(f"Could not open {root_file}")
    tree = f.Get("Analysis")

    horiz_pairs = [(0, 1, "C0|C1"), (1, 2, "C1|C2"), (3, 4, "C0|C1"),
                   (4, 5, "C1|C2"), (6, 7, "C0|C1"), (7, 8, "C1|C2")]
    vert_pairs = [(0, 3, "R0|R1"), (1, 4, "R0|R1"), (2, 5, "R0|R1"),
                  (3, 6, "R1|R2"), (4, 7, "R1|R2"), (5, 8, "R1|R2")]

    print("=== Vertical dividers (column boundary, crossing measured vs x) ===")
    results_v = []
    for a, b, tag in horiz_pairs:
        r = GRID[a][0]
        res = boundary_crossing(tree, a, b, "x", ROW_Y_CENTER[r], f"row{r}_{tag}_{a}{b}")
        results_v.append((a, b, r, tag, res))
        if res is None:
            print(f"  row{r} {tag}  pixel{a}-{b}: NO CROSSING FOUND")
        else:
            flag = "OK" if res["converged"] and res["jump"] > 10 else "UNCERTAIN"
            err_str = f"+/-{res['err']:.4f}" if res["err"] is not None else "(interp. only)"
            print(f"  row{r} {tag}  pixel{a}-{b}: x = {res['pos']:+.4f} {err_str}"
                  f"  jump={res['jump']:.1f}  [{flag}]")

    print("\n=== Horizontal dividers (row boundary, crossing measured vs y) ===")
    results_h = []
    for a, b, tag in vert_pairs:
        c = GRID[a][1]
        res = boundary_crossing(tree, a, b, "y", COL_X_CENTER[c], f"col{c}_{tag}_{a}{b}")
        results_h.append((a, b, c, tag, res))
        if res is None:
            print(f"  col{c} {tag}  pixel{a}-{b}: NO CROSSING FOUND")
        else:
            flag = "OK" if res["converged"] and res["jump"] > 10 else "UNCERTAIN"
            err_str = f"+/-{res['err']:.4f}" if res["err"] is not None else "(interp. only)"
            print(f"  col{c} {tag}  pixel{a}-{b}: y = {res['pos']:+.4f} {err_str}"
                  f"  jump={res['jump']:.1f}  [{flag}]")

    # --- plot: 3x3 grid of raw pixel amplitude maps ---
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleSize(0.07, "XY")
    ROOT.gStyle.SetLabelSize(0.06, "XY")
    ROOT.gStyle.SetTitleOffset(1.0, "X")
    ROOT.gStyle.SetTitleOffset(1.1, "Y")
    ROOT.gStyle.SetPalette(ROOT.kBird)

    canvas = ROOT.TCanvas("cpix", "Pixel maps", 1500, 1500)
    canvas.Divide(3, 3, 0.008, 0.008)

    keepalive = []
    for idx in range(9):
        r, c = GRID[idx]
        pad_num = r * 3 + c + 1  # R=0 (top row of pixels) drawn on top row of canvas
        pad = canvas.cd(pad_num)
        pad.SetLeftMargin(0.17)
        pad.SetRightMargin(0.15)
        pad.SetBottomMargin(0.15)

        h = build_profile2d(tree, z_expr(PIXEL_CHANNELS[idx]))
        h.SetTitle(f"pixel{idx}  ({','.join(map(str, PIXEL_CHANNELS[idx]))});x;y;mean sum")
        h.GetZaxis().SetRangeUser(0, 300)
        h.Draw("colz")
        keepalive.append(h)

    canvas.Update()
    canvas.SaveAs(out_png)
    print(f"\nSaved {out_png}")

    if sys.stdin.isatty():
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
