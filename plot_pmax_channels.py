#!/usr/bin/env python3
"""
pmax[1..16] spectra (one histogram per channel, 4x4 grid) for events inside
the sensor box determined in sensor_geometry_fit.py:
  -1.886 < 40*(x_pos1-x_pos2)+x_pos1 < -0.408
  -2.506 < 40*(y_pos1-y_pos2)+y_pos1 < -1.027

Usage:
  ./run_root_python.sh plot_pmax_channels.py [root_file] [output.png]
"""
import sys
import ROOT

DEFAULT_FILE = "/Volumes/DCRSD_2/TB11/stats_Run5_tracks.root"

X_LO, X_HI = -1.886, -0.408
Y_LO, Y_HI = -2.506, -1.027
CUT = (
    f"(40*(x_pos1-x_pos2)+x_pos1>{X_LO} && 40*(x_pos1-x_pos2)+x_pos1<{X_HI} && "
    f"40*(y_pos1-y_pos2)+y_pos1>{Y_LO} && 40*(y_pos1-y_pos2)+y_pos1<{Y_HI})"
)

NCHANNELS = 16
NBINS, XLO, XHI = 100, 0, 200


def main():
    root_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    out_png = sys.argv[2] if len(sys.argv) > 2 else "pmax_channels.png"

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        sys.exit(f"Could not open {root_file}")
    tree = f.Get("Analysis")

    ROOT.gStyle.SetOptStat(1110)
    ROOT.gStyle.SetTitleSize(0.08, "XY")
    ROOT.gStyle.SetLabelSize(0.07, "XY")
    ROOT.gStyle.SetTitleOffset(0.9, "X")
    ROOT.gStyle.SetTitleOffset(1.1, "Y")

    canvas = ROOT.TCanvas("cpmax", "pmax channels", 1600, 1200)
    canvas.Divide(4, 4, 0.001, 0.001)

    hists = []
    for ch in range(1, NCHANNELS + 1):
        pad = canvas.cd(ch)
        pad.SetLeftMargin(0.16)
        pad.SetBottomMargin(0.16)
        pad.SetLogy()

        hname = f"h_pmax{ch}"
        tree.Draw(f"pmax[{ch}]>>{hname}({NBINS},{XLO},{XHI})", CUT, "goff")
        h = ROOT.gDirectory.Get(hname)
        h.SetTitle(f"pmax[{ch}];pmax_{{{ch}}} [mV];events")
        h.SetLineColor(ROOT.kAzure + 2)
        h.SetFillColorAlpha(ROOT.kAzure + 2, 0.3)
        h.Draw("hist")
        hists.append(h)

    canvas.Update()
    canvas.SaveAs(out_png)
    print(f"Saved {out_png}")

    if sys.stdin.isatty():
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
