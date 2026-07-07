#!/usr/bin/env python3
"""
Sensor geometry (x/y limits + rotation) from the amplitude map built in
plot_amplitude_map.py: mean sum(pmax[1..16]) per event vs
(40*(x_pos1-x_pos2)+x_pos1, 40*(y_pos1-y_pos2)+y_pos1).

For each of the 4 sides of the sensor, the amplitude falls from the
in-sensor plateau (~pmax_tot ~ a few hundred mV) down to the out-of-sensor
floor (~100 mV) as an S-curve. We fit
    f(x) = p0 + p1*Erf((x-p2)/(sqrt(2)*p3))
so that p2 is the position where the amplitude crosses the 50% point of the
transition (offset p0), used as the edge/threshold reference (~125 mV,
per the user's rough pmax_tot>125 mV criterion).

To catch a possible small rotation of the sensor with respect to the (x,y)
axes, each side is fit multiple times in narrow slices running along that
side (e.g. the left/right edges are each fit in several y-slices). The
resulting edge position vs. slice-center is fit with a line; a nonzero
slope on the vertical edges (left/right) or horizontal edges (bottom/top)
directly gives the rotation angle of the sensor.

Usage:
  ./run_root_python.sh sensor_geometry_fit.py [root_file] [out.png]
"""
import sys
import math
import ROOT

DEFAULT_FILE = "/Volumes/DCRSD_2/TB11/stats_Run5_tracks.root"
NBINS_X, XLO, XHI = 170, -2.0, -0.3
NBINS_Y, YLO, YHI = 170, -2.6, -0.9

X_EXPR = "(40*(x_pos1-x_pos2)+x_pos1)"
Y_EXPR = "(40*(y_pos1-y_pos2)+y_pos1)"
Z_EXPR = "Sum$(pmax*(Iteration$>=1&&Iteration$<=16))"

# Narrow windows (in the edge's own coordinate) where each S-curve is fit.
LEFT_FIT = (-2.00, -1.80)
RIGHT_FIT = (-0.46, -0.34)
BOTTOM_FIT = (-2.60, -2.40)
TOP_FIT = (-1.15, -0.95)

# Ranges (along the direction parallel to each edge) sliced into N_SLICES
# independent S-curve fits, to probe for a rotation-induced trend.
LEFT_RIGHT_SLICE_RANGE = (-2.35, -1.15)  # sliced in y
BOTTOM_TOP_SLICE_RANGE = (-1.75, -0.55)  # sliced in x
N_SLICES = 6


def build_profile2d(tree):
    draw_str = (
        f"{Z_EXPR}:{Y_EXPR}:{X_EXPR}>>h({NBINS_X},{XLO},{XHI},{NBINS_Y},{YLO},{YHI})"
    )
    tree.Draw(draw_str, "", "prof goff")
    return ROOT.gDirectory.Get("h")


def fit_scurve(h1d, fit_lo, fit_hi, name):
    v_lo = h1d.GetBinContent(h1d.GetXaxis().FindBin(fit_lo))
    v_hi = h1d.GetBinContent(h1d.GetXaxis().FindBin(fit_hi))
    f = ROOT.TF1(name, "[0]+[1]*TMath::Erf((x-[2])/(TMath::Sqrt(2)*[3]))", fit_lo, fit_hi)
    f.SetParameters((v_lo + v_hi) / 2, (v_hi - v_lo) / 2, (fit_lo + fit_hi) / 2,
                     (fit_hi - fit_lo) / 6)
    status = int(h1d.Fit(f, "RQS").Status())
    return f, status


def slice_edges(h, slice_axis, slice_range, fit_range, n_slices, tag):
    """slice_axis='y' -> ProfileX in y-slices (left/right edges);
    slice_axis='x' -> ProfileY in x-slices (bottom/top edges)."""
    lo, hi = slice_range
    step = (hi - lo) / n_slices
    points = []
    for i in range(n_slices):
        s_lo, s_hi = lo + i * step, lo + (i + 1) * step
        center = (s_lo + s_hi) / 2
        if slice_axis == "y":
            b_lo = h.GetYaxis().FindBin(s_lo)
            b_hi = h.GetYaxis().FindBin(s_hi - 1e-6)
            h1d = h.ProfileX(f"{tag}_{i}", b_lo, b_hi)
        else:
            b_lo = h.GetXaxis().FindBin(s_lo)
            b_hi = h.GetXaxis().FindBin(s_hi - 1e-6)
            h1d = h.ProfileY(f"{tag}_{i}", b_lo, b_hi)
        f, status = fit_scurve(h1d, fit_range[0], fit_range[1], f"f_{tag}_{i}")
        pos, pos_err = f.GetParameter(2), f.GetParError(2)
        print(f"  [{tag}] slice center={center:+.3f}  edge={pos:+.4f} +/- {pos_err:.4f}"
              f"  sigma={f.GetParameter(3):.4f}  fitStatus={status}")
        points.append((center, pos, pos_err))
    return points


def fit_line(points, name):
    g = ROOT.TGraphErrors(len(points))
    for i, (c, pos, err) in enumerate(points):
        g.SetPoint(i, c, pos)
        g.SetPointError(i, 0.0, err)
    g.Fit("pol1", "QS")
    fpol = g.GetFunction("pol1")
    slope, slope_err = fpol.GetParameter(1), fpol.GetParError(1)
    intercept, intercept_err = fpol.GetParameter(0), fpol.GetParError(0)
    g.SetName(name)
    return g, slope, slope_err, intercept, intercept_err


def main():
    root_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    out_png = sys.argv[2] if len(sys.argv) > 2 else "sensor_geometry.png"

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        sys.exit(f"Could not open {root_file}")
    tree = f.Get("Analysis")

    h = build_profile2d(tree)

    print("Fitting left edge (x ~ -1.9), slicing in y:")
    left_pts = slice_edges(h, "y", LEFT_RIGHT_SLICE_RANGE, LEFT_FIT, N_SLICES, "left")
    print("Fitting right edge (x ~ -0.4), slicing in y:")
    right_pts = slice_edges(h, "y", LEFT_RIGHT_SLICE_RANGE, RIGHT_FIT, N_SLICES, "right")
    print("Fitting bottom edge (y ~ -2.5), slicing in x:")
    bottom_pts = slice_edges(h, "x", BOTTOM_TOP_SLICE_RANGE, BOTTOM_FIT, N_SLICES, "bottom")
    print("Fitting top edge (y ~ -1.05), slicing in x:")
    top_pts = slice_edges(h, "x", BOTTOM_TOP_SLICE_RANGE, TOP_FIT, N_SLICES, "top")

    g_left, m_left, dm_left, b_left, db_left = fit_line(left_pts, "g_left")
    g_right, m_right, dm_right, b_right, db_right = fit_line(right_pts, "g_right")
    g_bottom, m_bot, dm_bot, b_bot, db_bot = fit_line(bottom_pts, "g_bottom")
    g_top, m_top, dm_top, b_top, db_top = fit_line(top_pts, "g_top")

    # x0(y) = m*y + b for left/right -> slope = -tan(theta)
    # y0(x) = m*x + b for bottom/top -> slope = +tan(theta)
    theta_left = -math.atan(m_left)
    theta_right = -math.atan(m_right)
    theta_bottom = math.atan(m_bot)
    theta_top = math.atan(m_top)

    dtheta_left = dm_left / (1 + m_left ** 2)
    dtheta_right = dm_right / (1 + m_right ** 2)
    dtheta_bottom = dm_bot / (1 + m_bot ** 2)
    dtheta_top = dm_top / (1 + m_top ** 2)

    thetas = [theta_left, theta_right, theta_bottom, theta_top]
    dthetas = [dtheta_left, dtheta_right, dtheta_bottom, dtheta_top]
    weights = [1 / e ** 2 for e in dthetas]
    theta_avg = sum(t * w for t, w in zip(thetas, weights)) / sum(weights)
    theta_avg_err = 1 / math.sqrt(sum(weights))

    y_ref = sum(LEFT_RIGHT_SLICE_RANGE) / 2
    x_ref = sum(BOTTOM_TOP_SLICE_RANGE) / 2
    x_min = m_left * y_ref + b_left
    x_max = m_right * y_ref + b_right
    y_min = m_bot * x_ref + b_bot
    y_max = m_top * x_ref + b_top

    print()
    print("=== Edge lines (position = slope * slice_center + intercept) ===")
    print(f"  left   x0(y) : slope={m_left:+.5f}+/-{dm_left:.5f}  intercept={b_left:+.4f}+/-{db_left:.4f}")
    print(f"  right  x0(y) : slope={m_right:+.5f}+/-{dm_right:.5f}  intercept={b_right:+.4f}+/-{db_right:.4f}")
    print(f"  bottom y0(x) : slope={m_bot:+.5f}+/-{dm_bot:.5f}  intercept={b_bot:+.4f}+/-{db_bot:.4f}")
    print(f"  top    y0(x) : slope={m_top:+.5f}+/-{dm_top:.5f}  intercept={b_top:+.4f}+/-{db_top:.4f}")
    print()
    print("=== Rotation angle estimates (theta, per edge) ===")
    print(f"  from left edge  : {math.degrees(theta_left):+.4f} deg")
    print(f"  from right edge : {math.degrees(theta_right):+.4f} deg")
    print(f"  from bottom edge: {math.degrees(theta_bottom):+.4f} deg")
    print(f"  from top edge   : {math.degrees(theta_top):+.4f} deg")
    print(f"  weighted average: {math.degrees(theta_avg):+.4f} +/- {math.degrees(theta_avg_err):.4f} deg")
    print()
    print(f"=== Sensor box (evaluated at y={y_ref:.2f} for x-edges, x={x_ref:.2f} for y-edges) ===")
    print(f"  x_min = {x_min:+.4f}   x_max = {x_max:+.4f}   width  = {x_max - x_min:.4f}")
    print(f"  y_min = {y_min:+.4f}   y_max = {y_max:+.4f}   height = {y_max - y_min:.4f}")

    # --- plot: 2D map with fitted edges overlaid, + 4 slice-position graphs ---
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetTitleSize(0.055, "XY")
    ROOT.gStyle.SetLabelSize(0.045, "XY")
    ROOT.gStyle.SetTitleOffset(1.15, "X")
    ROOT.gStyle.SetTitleOffset(1.3, "Y")

    canvas = ROOT.TCanvas("cgeom", "Sensor geometry", 1500, 950)
    canvas.Divide(3, 2, 0.012, 0.012)

    pad1 = canvas.cd(1)
    pad1.SetLeftMargin(0.16)
    pad1.SetRightMargin(0.16)
    pad1.SetBottomMargin(0.14)
    h2 = build_profile2d(tree)
    h2.SetTitle("Amplitude map with fitted edges;x;y;mean #sum pmax")
    h2.GetZaxis().SetRangeUser(0, 300)
    h2.GetZaxis().SetTitleOffset(1.5)
    h2.GetZaxis().SetLabelSize(0.04)
    h2.GetZaxis().SetTitleSize(0.045)
    ROOT.gStyle.SetPalette(ROOT.kBird)
    h2.Draw("colz")
    lines = []
    for (m, b, rng, vertical) in [
        (m_left, b_left, LEFT_RIGHT_SLICE_RANGE, True),
        (m_right, b_right, LEFT_RIGHT_SLICE_RANGE, True),
        (m_bot, b_bot, BOTTOM_TOP_SLICE_RANGE, False),
        (m_top, b_top, BOTTOM_TOP_SLICE_RANGE, False),
    ]:
        p0, p1 = rng
        if vertical:
            ln = ROOT.TLine(m * p0 + b, p0, m * p1 + b, p1)
        else:
            ln = ROOT.TLine(p0, m * p0 + b, p1, m * p1 + b)
        ln.SetLineColor(ROOT.kRed)
        ln.SetLineWidth(2)
        ln.Draw()
        lines.append(ln)

    graphs = [
        (2, g_left, "y  (left edge)", "fitted x0"),
        (3, g_right, "y  (right edge)", "fitted x0"),
        (5, g_bottom, "x  (bottom edge)", "fitted y0"),
        (6, g_top, "x  (top edge)", "fitted y0"),
    ]
    for pad_num, g, xtitle, ytitle in graphs:
        pad = canvas.cd(pad_num)
        pad.SetLeftMargin(0.18)
        pad.SetBottomMargin(0.16)
        g.SetMarkerStyle(20)
        g.GetXaxis().SetTitle(xtitle)
        g.GetYaxis().SetTitle(ytitle)
        g.GetXaxis().SetTitleSize(0.055)
        g.GetYaxis().SetTitleSize(0.055)
        g.GetXaxis().SetLabelSize(0.045)
        g.GetYaxis().SetLabelSize(0.045)
        g.GetXaxis().SetTitleOffset(1.15)
        g.GetYaxis().SetTitleOffset(1.5)
        g.Draw("AP")
        g.GetFunction("pol1").Draw("same")

    canvas.Update()
    canvas.SaveAs(out_png)
    print(f"\nSaved {out_png}")

    if sys.stdin.isatty():
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
