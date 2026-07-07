#!/usr/bin/env python3
"""
pmax[channel] distribution for all 16 channels, each restricted to the
region(s) of the pixel(s) that channel belongs to (a channel can be shared
by up to 4 neighbouring pixels, e.g. channel 15 is in both pixel0 and
pixel1 -- the selection is the union of all such pixel boxes). Pixel boxes
come from pixel_geometry_fit.py's boundary fit. Each distribution is fit
with a Landau function, with the fit stat box shown on the plot.

Usage:
  ./run_root_python.sh plot_amplitude_map.py [root_file] [output.png]
"""
import sys
import ROOT

DEFAULT_FILE = "/Volumes/DCRSD_2/TB11/stats_Run5_tracks.root"

X_EXPR = "(40*(x_pos1-x_pos2)+x_pos1)"
Y_EXPR = "(40*(y_pos1-y_pos2)+y_pos1)"

PIXEL_CHANNELS = {
    0: (14, 15, 8, 9), 1: (15, 13, 10, 8), 2: (13, 16, 11, 10),
    3: (9, 8, 6, 7), 4: (8, 10, 5, 6), 5: (10, 11, 4, 5),
    6: (7, 6, 2, 3), 7: (6, 5, 1, 2), 8: (5, 4, 12, 1),
}
# (x_lo, x_hi, y_lo, y_hi), from pixel_geometry_fit.py's boundary crossings.
PIXEL_BOXES = {
    0: (-1.886, -1.391, -1.523, -1.027), 1: (-1.391, -0.903, -1.523, -1.027),
    2: (-0.903, -0.408, -1.523, -1.027), 3: (-1.886, -1.391, -2.014, -1.523),
    4: (-1.391, -0.903, -2.014, -1.523), 5: (-0.903, -0.408, -2.014, -1.523),
    6: (-1.886, -1.391, -2.506, -2.014), 7: (-1.391, -0.903, -2.506, -2.014),
    8: (-0.903, -0.408, -2.506, -2.014),
}

NCHANNELS = 16
NBINS, XLO, XHI = 100, 0, 200


def channel_to_pixels(channel):
    return sorted(idx for idx, chans in PIXEL_CHANNELS.items() if channel in chans)


def build_cut(pixels):
    parts = []
    for idx in pixels:
        x_lo, x_hi, y_lo, y_hi = PIXEL_BOXES[idx]
        parts.append(f"({X_EXPR}>{x_lo} && {X_EXPR}<{x_hi} && {Y_EXPR}>{y_lo} && {Y_EXPR}<{y_hi})")
    return "(" + " || ".join(parts) + ")"


def main():
    root_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    out_png = sys.argv[2] if len(sys.argv) > 2 else "pmax_all_channels_pixel_region.png"

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        sys.exit(f"Could not open {root_file}")
    tree = f.Get("Analysis")

    ROOT.gStyle.SetOptStat(1110)
    ROOT.gStyle.SetOptFit(1111)
    ROOT.gStyle.SetTitleSize(0.08, "XY")
    ROOT.gStyle.SetLabelSize(0.07, "XY")
    ROOT.gStyle.SetTitleOffset(0.9, "X")
    ROOT.gStyle.SetTitleOffset(1.1, "Y")
    ROOT.gStyle.SetStatFontSize(0.06)

    canvas = ROOT.TCanvas("cpmax", "pmax channels (pixel region, Landau fit)", 1800, 1400)
    canvas.Divide(4, 4, 0.001, 0.001)

    hists, fits = [], []
    for ch in range(1, NCHANNELS + 1):
        pixels = channel_to_pixels(ch)
        cut = build_cut(pixels)

        pad = canvas.cd(ch)
        pad.SetLeftMargin(0.16)
        pad.SetBottomMargin(0.16)
        pad.SetLogy()

        hname = f"h_pmax{ch}"
        tree.Draw(f"pmax[{ch}]>>{hname}({NBINS},{XLO},{XHI})", cut, "goff")
        h = ROOT.gDirectory.Get(hname)
        pix_str = ",".join(str(p) for p in pixels)
        h.SetTitle(f"pmax[{ch}], pixel(s) {pix_str};pmax_{{{ch}}} [mV];events")
        h.SetLineColor(ROOT.kAzure + 2)
        h.SetFillColorAlpha(ROOT.kAzure + 2, 0.3)

        fit = ROOT.TF1(f"flandau{ch}", "landau", XLO, XHI)
        fit.SetLineColor(ROOT.kRed)
        h.Fit(fit, "Q")

        h.Draw("hist")
        fit.Draw("same")

        print(f"pmax[{ch}] pixel(s) {pixels}: MPV={fit.GetParameter(1):.2f}"
              f"  sigma={fit.GetParameter(2):.2f}  chi2/ndf={fit.GetChisquare():.1f}/{fit.GetNDF()}")

        hists.append(h)
        fits.append(fit)

    canvas.Update()
    canvas.SaveAs(out_png)
    print(f"Saved {out_png}")

    if sys.stdin.isatty():
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
