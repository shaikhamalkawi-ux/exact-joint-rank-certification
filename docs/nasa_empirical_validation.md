# NASA Battery empirical EJRC extension

This branch contains a direct-from-raw empirical uncertainty audit using the official NASA PCoE Battery Data Set.

## Why B0006 and B0018

The dataset README describes batteries 6 and 18 at room temperature, discharged at 2 A to the same 2.5 V cutoff. Batteries 5 and 7 use different cutoff voltages, so they are intentionally excluded from this direct pairwise decision demonstration.

## Direct source lock

The workflow downloads the official NASA archive from:

https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

Locked archive SHA-256:

`82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`

Raw MAT files extracted from the official bundle:

- B0006 SHA-256 `fa818ab4db5db8ab21e910b6dd6c3e20d3761bb9672089e1a4de8f96074616c5`
- B0018 SHA-256 `d1e6c923a43ea1c9666b3a90bbb521757a067fd60b4d17dbfaa49c50b179da69`

Their Git blob SHA-1 identifiers match independently observed public mirrors.

## Measurement-derived criteria

Each discharge cycle supplies four criteria:

1. capacity, benefit;
2. mean discharge voltage, benefit;
3. mean measured temperature, cost;
4. absolute mean-current tracking error from the 2 A target, cost.

The pooled discharge measurements from B0006 and B0018 are converted to strictly interior `(0,1)` scores using empirical average ranks divided by `N+1`, with reversal for cost criteria. Equal reference weights `(0.25,0.25,0.25,0.25)` are used as a neutral analytical choice; they are not learned from the NASA measurements.

Early, mid, and late windows are centered at 20%, 50%, and 80% of each cell's observed discharge trajectory, using +/-5 percentage-point windows. The median score is the nominal stage value, while observed min/max scores form the empirical score box.

## Locked findings

- **Early:** B0006 is nominally higher, but the empirical-box certificate is negative (`C=-0.769651`), so the nominal early-life winner is not universally certified.
- **Mid:** B0018 is nominally higher and certified (`C=+0.100841`). Incorporating the empirical score box, the B0018>B0006 relation remains certified under symmetric multiplicative weight uncertainty up to `r*=0.414379` around equal weights.
- **Late:** B0018 is nominally higher and certified (`C=+0.164257`). Its lower empirical score bound exceeds B0006's upper bound on all four criteria in the locked window, giving componentwise empirical interval dominance and therefore independence from the equal-weight choice for any strictly positive weights.

A window-width audit using full life windows of 6%, 8%, 10%, 15%, and 20% preserves the same qualitative pattern: early is not certified, while mid and late are certified.

## Claim boundary

This is a measurement-derived uncertainty demonstration under the published NASA protocol. It is not a universal claim of intrinsic battery superiority. The empirical bounds are observed ranges, not confidence intervals. The rectangular score box is conservative because criterion extrema can come from different discharge cycles.

The main paper should not be changed until an editorial scope decision is made about whether this validation belongs in the main text or supplementary material.
