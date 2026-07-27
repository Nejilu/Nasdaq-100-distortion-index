# ASML anomaly in IQQ and CNDX snapshots

## Technical summary

ASML's 49.13% free-float weight did not originate in the IQQ or CNDX holdings.
Both calculations reused an inconsistent yfinance `floatShares` value of
21,331,633,667 shares. The added validation now excludes this observation
without substituting `sharesOutstanding`.

## One erroneous observation affected both universes

Snapshots 10 (IQQ) and 11 (CNDX) recorded the same ASML price of USD 1,747.58,
the same float of 21,331,633,667 shares, and the same 49.13% free-float weight.
ASML's published weight remained close to 0.74% in both ETFs. This symmetry
located the defect in the shared market-data source rather than in the two
independent holdings sources.

## Consistency checks invalidate the ASML float

The live yfinance response from July 18, 2026 provided all of the following:

- `floatShares`: 21,331,633,667
- `sharesOutstanding`: 384,100,000
- `marketCap`: USD 671,245,467,648
- Price: USD 1,747.58

The reported float was 55.5 times shares outstanding. Price multiplied by float
implied approximately USD 37.28 trillion, also 55.5 times the published total
market capitalization. These checks are descriptive: they identify a unit or
provider inconsistency without determining the correct float value.

## Added rejection rules

An observation is now marked `invalid_float_inconsistent` and excluded when its
float exceeds shares outstanding by more than 10%, or when its implied
free-float capitalization exceeds total market capitalization by more than 25%.
The tolerances allow for reporting-date differences between fields. No fallback
to 100% of shares outstanding is performed.

The same validation showed that yfinance reports one identical consolidated
Alphabet float for GOOG and GOOGL. When several classes have the exact same
float, similar prices and market capitalizations, and a float consistent with
their combined shares outstanding, the total is allocated in proportion to
class-level shares outstanding. The rows remain separate and are marked
`valid_shared_float_allocated`.

## Limitations and robustness

The checks still depend on yfinance fields that may be missing or stale. Without
those fields, a positive float cannot be validated by these two rules.
`coverage_ratio`, snapshot status, and the invalid-float count must therefore
remain visible.

## End-to-end validation

Final live snapshots 14 (IQQ) and 15 (CNDX) had coverage of 98.7870% and
98.7877%, respectively. ASML was marked `invalid_float_inconsistent` and had no
`float_weight`. GOOG and GOOGL were marked `valid_shared_float_allocated`,
remained separate rows, and both the published and free-float weights of the 100
valid rows summed to 1. AAPL had the highest free-float weight at 14.04%.

Earlier contaminated snapshots and snapshots created during diagnosis were
retained in SQLite with the `invalidated_data_quality` status, but excluded from
the dashboard history chart.

## Current ACWI-based treatment

The primary free-float reference now comes from the official iShares ACWI
holdings. ACWI lists `ASML` on Euronext Amsterdam, while the Nasdaq-100 ETF
position is explicitly an ADR. The two rows must not be treated as the same
security for float purposes.

`ASML`, `ARM`, and `PDD` are therefore labeled `ADR/ADS` and bypass ACWI. They
use the calibrated yfinance fallback path. ASML remains excluded with
`invalid_yfinance_fallback` because the inconsistency documented above is still
present. This is explicit in each component's `security_type`,
`reference_source`, and `data_status`.

The remaining open data problem is a reliable ADR-specific float source. No
primary-listing ACWI value or total shares-outstanding value is substituted
automatically.
