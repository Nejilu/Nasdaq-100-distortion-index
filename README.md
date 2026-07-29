# NDX Weight Distortion Index

This project measures the gap between published Nasdaq-100 ETF weights and the
weights the same securities would receive under a pure capitalization-based
counterfactual.

The default view uses the official iShares ACWI portfolio as a free-float
reference:

```text
reference_mass = ACWI holding market value
float_weight = reference_mass / sum(reference_mass for Nasdaq-100 constituents)
weight_delta = actual_weight - float_weight
NDX_WDI = 50 x sum(abs(weight_delta))
```

ACWI market values are used instead of the displayed `Weight (%)` field because
the market values retain precision for small positions. ADR/ADS securities and
names absent from ACWI use a yfinance free-float fallback calibrated into the
same ACWI fund-value units with the median ratio observed across matched names.
The Nasdaq-listed `ASML` ADR is a maintained exception: its float is fixed at
`88,000,000` ADRs because yfinance's consolidated value is not valid for that
listing. Its listed total share count is also fixed at `88,000,000`, preventing
the annual modified-cap calculation from deriving an unsupported `3x` ratio.

```text
fallback_scale = median(ACWI_market_value / yfinance_float_market_cap)
fallback_reference_mass = yfinance_float_market_cap x fallback_scale
counterfactual_weight = reference_mass / sum(all_ACWI_and_fallback_reference_masses)
```

The last line renormalizes the covered selection to exactly 100%; fallbacks are
never normalized separately from the ACWI-matched securities.

The dashboard can also switch to total listed capitalization:

```text
total_market_cap = price x shares_outstanding
total_cap_weight = total_market_cap / sum(total_market_cap)
weight_delta = actual_weight - total_cap_weight
```

An `NDX_WDI` of 5 means that 5% of the total index weight would need to be
reallocated to move from one distribution to the other.

## Scope and methodology

- Two independent universes are stored: `non_ucits` and `ucits`. They are never
  merged or averaged.
- Two counterfactual bases are stored separately in SQLite: `float` and `total`.
  Switching the dashboard control never mixes their snapshots or histories.
- Published ETF weights are always used as reported by the selected fund. The
  project does not reconstruct ETF weights from prices or quantities.
- GOOG and GOOGL remain separate securities.
- Cash and explicitly non-equity positions are removed before equity weights are
  normalized to 100%.
- ACWI holding market values come from the official BlackRock/iShares public
  holdings download. Exact tickers are disambiguated with company names, which
  prevents collisions such as US `ADP` versus Aéroports de Paris and US `ROP`
  versus Roche.
- ADR/ADS rows are labeled and deliberately bypass ACWI because ACWI may hold
  the issuer's primary listing rather than the US depositary receipt. Current
  defaults are `ARM`, `ASML`, and `PDD`; `NDX_ADR_TICKERS` can extend the list.
- Prices, `floatShares`, `sharesOutstanding`, and `marketCap` come from
  `yfinance`. They support total capitalization, fallback weights, and fallback
  consistency checks. The `ASML` ADR overrides both `floatShares` and
  `sharesOutstanding` with `88,000,000`; the float source is persisted as
  `hardcoded_float_override`.
- The internal yfinance SQLite cache is stored in `data/yfinance_cache` so the
  application remains usable when the Windows user cache is not writable.
- An ACWI match does not require a yfinance price or float count. A missing or
  invalid yfinance value only excludes a security that requires fallback.
- A fallback float inconsistent with shares outstanding or total market
  capitalization is excluded without substitution.
- The legacy all-yfinance method remains the global fallback if the ACWI
  download cannot be validated, but every fallback observation passes the same
  float/share-count and float-cap/market-cap consistency checks.
- `coverage_ratio` measures the published weight represented before exclusions.
  Covered published weights are then normalized to 100% so both distributions
  are compared over the same investable set.
- `complete` means coverage is at least `NDX_COVERAGE_THRESHOLD` (99% by
  default); lower coverage is reported as `partial_coverage`. An all-yfinance
  reference is explicitly reported as `degraded_fallback` or
  `degraded_partial_coverage`, regardless of its nominal coverage.

The ETF holdings remain proxies for the index, and free market-data sources are
not official or guaranteed. Local history starts with the first saved snapshot.

## NDX vs S&P 500 Active Share

The second dashboard panel compares the selected Nasdaq-100 ETF with the
matching iShares S&P 500 ETF:

| Universe | Nasdaq-100 proxy | S&P 500 proxy |
| --- | --- | --- |
| Non-UCITS | IQQ | IVV |
| UCITS | CNDX | CSPX |

Each pair is normalized independently over its published equity holdings. The
calculation then uses the full union of securities in both funds, assigning
zero to a security that is absent from one side:

```text
active_share = 50 x sum(abs(NDX_weight - SPX_weight))
```

An Active Share of 46.7% means that 46.7% of either portfolio would need to be
reallocated for the two normalized weight distributions to match. It measures
weight and membership differences between the ETF proxies; it is not a return,
tracking-error, or performance forecast.

The panel includes:

- the ten largest NDX overweights and ten largest S&P 500 overweights;
- a Top-X slider that compares the aggregate weight of the largest NDX names in
  both portfolios;
- an overlaid constituent breakdown for that selected Top-X set;
- three synthetic portfolios built from the full holdings union: the positive
  NDX differences, the positive S&P 500 differences, and their shared overlap,
  each independently renormalized to 100%;
- an annual-reconstitution toggle that replaces current NDX weights with the
  same public-data annual simulation used by the distortion panel, while the
  S&P 500 reference remains unchanged.

```text
NDX_active_holding = max(NDX_weight - SPX_weight, 0) / active_share
SPX_active_holding = max(SPX_weight - NDX_weight, 0) / active_share
overlap_holding = min(NDX_weight, SPX_weight) / (1 - active_share)
```

Each elongated synthetic chart shows its 25 largest holdings. The remaining
`Other` weight is reported below the chart instead of being drawn as a bar. An
expandable table below each chart exposes its 50 largest holdings with rank,
company name, normalized weight, and cumulative weight.

Official iShares holdings downloads are the primary S&P 500 sources. Optional
local files can be configured with `NON_UCITS_SPX_HOLDINGS_CSV` and
`UCITS_SPX_HOLDINGS_CSV`.

## If rebalanced today

Each live snapshot also calculates an `NDX_WDI` using the weights that the
Nasdaq-100 would receive if its full annual December reconstitution used the
snapshot date's prices and public inputs.
The implementation follows the
[Nasdaq-100 methodology effective May 1, 2026](https://indexes.nasdaqomx.com/docs/Methodology_NDX.pdf)
and the official
[Nasdaq index weight calculation rules](https://indexes.nasdaqomx.com/docs/Nasdaq_Index_Weight_Calculations.pdf).

The daily pipeline:

1. Builds the eligible universe from the official Nasdaq stock screener and
   Nasdaq Trader symbol directory.
2. Groups multiple share classes at company level with SEC CIK identifiers.
3. Applies Nasdaq listing tier, non-financial, security-type, three-month ADVT,
   seasoning, and the annual top-75/100/125 constituent-selection sequence.
   Non-constituent ADRs are excluded unless primary-listing and listed
   depositary-share inputs can be verified. Current membership is a conservative
   public proxy for the prior-top-100 and post-reconstitution-addition flags.
4. Recalculates every security's initial weight from modified capitalization:

   ```text
   acwi_conversion_scale =
       90th percentile(ACWI_float_mass / listed_total_cap)

   converted_total_mass = listed_total_cap x acwi_conversion_scale
   modified_cap_mass = min(converted_total_mass, 3 x ACWI_float_mass)
   ```

   The upper-quantile calibration estimates the common fund-value scale from
   ACWI names whose free float is close to 100%. Direct ACWI matches never use
   yfinance `floatShares`. For ADR/ADS securities or absent ACWI positions,
   yfinance free float remains a documented fallback, except for the maintained
   `ASML` ADR override of `88,000,000` floating receipts.
5. Aggregates securities by company and iterates the annual company constraints:
   a company above 24% is reduced to at most 20%; if companies above 4.5%
   aggregate to at least 48%, that cohort is reduced to 40%. Initial rank is
   preserved.
6. Returns to security weights and iterates the annual security constraints:
   a security above 15% is reduced to at most 14%; if the five largest
   securities aggregate to at least 40%, they are reduced to 38.5% and every
   security outside the top five is capped at the lower of 4.4% or the
   fifth-largest weight.
7. Converts the final weights into the proportional Index Shares represented by
   the simulation.
8. Compares the resulting security weights with the selected free-float or
   total-cap reference and recalculates `NDX_WDI`.

The dashboard displays this score beside the live reading. Enabling **Show
annual-reconstitution weights** switches the constituent rankings and main
difference chart to simulated weights, adds signed changes versus current
weights, and shows dedicated charts for the largest weight movements and
simulated index entries/exits.

The weight constraints are deterministic, but the composition result is a
public-data simulation rather than an official Nasdaq review. Nasdaq retains
discretion and does not publish every review input. The snapshot records
`rebalance_status`, source, coverage, additions, removals, fallbacks, and notes
so this limitation remains visible through SQLite and the API.

### Annual reconstitution dashboard

The dedicated **Annual Reconstitution** panel reconstructs every weighting
stage from the persisted live snapshot and makes the annual rules auditable:

- observed company and security concentrations, their distance to each trigger,
  and the corresponding Nasdaq target;
- a ranked company-level view of the 4.5% cohort, including its cumulative
  weight against the 48% trigger and 40% adjustment target;
- cumulative selected-company Modified Market Capitalization before and after
  company capping;
- the Modified Market Capitalization multiple relative to the ACWI free-float
  input, including the `3x` ceiling;
- company-stage, security-stage, and total redistributed weight;
- cumulative transfers by initial rank, the largest donors and recipients, and
  rank preservation measured separately at each official capping stage;
- current versus simulated annual weights, plus simulated index entries and
  exits.

An audit expander exposes the reconstructed stage values and residuals. The
snapshot does not persist unselected candidates ranked 101-125, so the panel
does not infer or display a fictitious distance to those selection cutoffs. It
shows the selected 100-company distribution and the simulated membership
changes instead.

## Capitalization references

The two dashboard bases are analytical comparison scenarios:

- **Free Float** uses shares readily available for public trading. This is the
  convention used by major investable benchmarks such as the
  [S&P 500](https://www.spglobal.com/spdji/en/methodology/article/sp-us-indices-methodology/)
  and the
  [MSCI World](https://www.msci.com/indexes/index/990100/msci-world-index).
- **Total** uses all shares outstanding, including strategic holdings that may
  not be readily tradable. The
  [Nasdaq Composite](https://www.nasdaq.com/newsroom/nasdaq-composite-vs-nasdaq-100-what-investors-should-know)
  is a useful reference for this approach.

The live free-float and total views remain analytical reference scenarios. The
separate **If rebalanced today** calculation applies Nasdaq's modified
capitalization and concentration rules before comparing those simulated index
weights with the selected reference.

## Holdings sources and fallbacks

Every source must contain between 90 and 130 equities, unique tickers, and valid
published weights. Top-10 pages, HTML responses, and incomplete exports are
rejected.

`non_ucits` source order:

1. Explicit local CSV (`NON_UCITS_HOLDINGS_CSV` or `--holdings-csv`)
2. Official BlackRock/iShares IQQ download
3. Public Invesco QQQ holdings
4. CSV URLs configured in `NON_UCITS_FALLBACK_URLS`

`ucits` source order:

1. Explicit local CSV (`UCITS_HOLDINGS_CSV` or `--holdings-csv`)
2. Official iShares CNDX CSV
3. Public Invesco EQQQ holdings
4. CSV URLs configured in `UCITS_FALLBACK_URLS`, including optional Xtrackers
   or UBS sources

The observed public Nasdaq page exposes only leading holdings and is not accepted
as a complete universe. Each snapshot retains the selected `reference_fund`,
published holdings date, and failures from preceding sources.

For the free-float calculation, the official ACWI download is then used as the
primary reference. Matching is one Nasdaq-100 ticker at a time and requires a
compatible company name. The resulting ACWI market values, plus any calibrated
yfinance fallback values, are renormalized to 100% across the covered
Nasdaq-100 selection.

## Quarterly SEC history

The dashboard's long-term chart is not built from local refresh snapshots. It
contains exactly one observation per public quarter from the first available
Form N-PORT data in September 2019.

For each quarter:

1. The QQQ and SPGM N-PORT-P accessions are discovered in SEC EDGAR.
2. Equity positions are matched by exact CUSIP, not ticker or company name.
3. Missing ADRs and incompatible listing forms are excluded from both
   distributions.
4. For each other QQQ security absent from SPGM, its counterfactual weight is
   estimated from the median observed ratio among matched QQQ overweights:
   `estimated_SPGM_weight = QQQ_weight / median_overweight_ratio`.
5. Observed and estimated SPGM weights are combined, then QQQ and SPGM are each
   independently renormalized to 100%.
6. `NDX_WDI = 50 x sum(abs(QQQ_weight - SPGM_weight))`.

This historical method is deliberately separate from the live ACWI method. SPGM
did not hold every QQQ security in the early years: September 2019 matches 53 of
103 QQQ equity CUSIPs and covers about 85% of the comparable QQQ equity weight.
The matched-only score is retained as `ndx_wdi_raw`, while `ndx_wdi` contains
the corrected series used by the chart.

The correction was selected through historical masking tests rather than from a
single current snapshot. Rank-based missingness patterns observed from 2019
through 2021 were applied to the better-covered 2022-2026 quarters. Across 170
tests at 86.5% average coverage, median-ratio imputation reduced mean absolute
error from 2.40 to 0.65 NDX_WDI points (72.7%) and improved the result in 169
tests. The raw score, correction size, estimated count, and non-comparable
exclusion count remain available for audit.

Rebuild the complete history with:

```powershell
python run_quarterly_history.py
```

The command stores all 54 complete filing XML documents under
`data/sec_nport_filings/raw/{qqq|spgm}/{report_date}/{accession}/`, plus:

- `manifest.csv`: accession, official SEC URL, local path, transport, file size,
  and SHA-256 checksum.
- `positions.csv.gz`: every normalized equity position from both funds.
- `constituent_history.csv.gz`: raw and corrected quarterly weights, weight gap,
  contribution, estimation status, and exclusion status for every QQQ CUSIP.
- `data/edgar_quarterly_history.csv`: the compact 27-point dashboard series.

The full local filing archive is ignored by Git. Direct SEC archive downloads
are attempted first. When SEC blocks the runtime's network address, the
configured reader transport retrieves the complete XML document element from
the same official SEC archive URL; the manifest records which transport was
used.

## Architecture

```text
qqq_holdings_provider.py  # Nasdaq-100 and S&P 500 iShares provider chains
acwi_weights_provider.py  # ACWI matching, ADR labels, calibrated fallbacks
active_share.py           # NDX/SPX union, normalization, and Active Share
nasdaq100_rebalance.py    # annual/quarterly selection and weighting engine
rebalance_analytics.py    # annual thresholds, stages, transfers, and rank audit
edgar_quarterly_history.py # N-PORT archive, CUSIP history, quarterly scores
market_data_provider.py   # live market data through yfinance
distortion_engine.py      # pure calculations, coverage, and statuses
database.py               # SQLite schema and access
snapshot_service.py       # live snapshot orchestration
api.py                    # FastAPI application
dashboard.py              # Streamlit dashboard
run_snapshot.py           # one-off and scheduled CLI
run_quarterly_history.py  # rebuild the SEC quarterly archive and chart data
tests/                    # calculations, parsing, persistence, and API tests
```

Providers expose small contracts, allowing a licensed or more reliable data feed
to replace Invesco or yfinance without changing the calculation engine, API, or
dashboard.

## Installation

Python 3.11 or 3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS and Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` locks all direct and transitive versions for reproducible
Python 3.11 and 3.12 installations. Direct dependency ranges are maintained in
`requirements.in`. Regenerate the lock after changing those ranges or when
preparing a controlled dependency update:

```bash
uv pip compile requirements.in --universal --python-version 3.11 \
  --upgrade -o requirements.txt
python -m pytest
```

## First run

Create live snapshots, start the API, and then start the dashboard:

```bash
python run_snapshot.py --universe all
python -m uvicorn api:app --reload
```

In a second terminal:

```bash
python -m streamlit run dashboard.py
```

- Interactive API: `http://127.0.0.1:8000/docs`
- Dashboard: `http://localhost:8501`

Both services can instead be launched as detached background processes with
the active Python environment:

```bash
python run_local.py
```

The launcher returns immediately, prints the process IDs, and skips services
whose ports are already listening. It uses detached Windows processes or a new
POSIX session as appropriate. Runtime output is written under `data/`.

The dashboard header selects either `NDX Distortion Index` or
`NDX vs S&P 500`. Both panels provide a matching universe control:

- `Non-UCITS` or `UCITS`

The distortion panel also provides `Free Float` or `Total`. Active Share uses
published ETF weights and therefore does not expose that counterfactual basis
control.

The Refresh button updates the selected universe and capitalization basis using
live data only.

## Live-source usage

```bash
# Both universes; fails explicitly if no provider chain succeeds
python run_snapshot.py --universe all

# Both universes using price x shares outstanding
python run_snapshot.py --universe all --basis total

# Attach an explicit local holdings CSV to one universe
python run_snapshot.py --universe non_ucits --holdings-csv path/holdings_qqq.csv
python run_snapshot.py --universe ucits --holdings-csv path/holdings_cndx.csv
```

All URLs are configurable through `.env`. A compatible CSV must contain at least
a ticker and weight, and should ideally include company name and asset class.
There is no fallback to synthetic data.

## API

```text
GET  /api/current
GET  /api/current?universe=ucits
GET  /api/current?universe=ucits&weighting_basis=total
GET  /api/history?limit=365&universe=non_ucits
GET  /api/components?universe=ucits&weighting_basis=total&ranking=contributors&limit=20
GET  /api/active-share?universe=non_ucits&ranking=contributors&limit=20
GET  /api/active-share?universe=ucits&rebalanced=true
POST /api/recompute
```

Example recomputation:

```bash
curl -X POST http://127.0.0.1:8000/api/recompute \
  -H "Content-Type: application/json" \
  -d '{"universe":"all","weighting_basis":"total"}'
```

For `/api/components`, `ranking` accepts `all`, `overweights`,
`underweights`, or `contributors`. For `/api/active-share`, it accepts `all`,
`ndx_overweights`, `spx_overweights`, or `contributors`.

## Daily snapshots

The built-in loop waits until the selected local time and remains active:

```bash
python run_snapshot.py --universe all --daily --at 18:00
```

In production, prefer a system scheduler for the one-off command:

```cron
0 18 * * 1-5 cd /path/to/ndx-wdi && .venv/bin/python run_snapshot.py --universe all
```

On Windows, create a Task Scheduler entry that runs
`C:\path\.venv\Scripts\python.exe` with the arguments
`run_snapshot.py --universe all` and uses the project directory as its working
directory.

## Tests

```bash
python -m pytest
```

The reference case compares published weights `A=50%`, `B=30%`, and `C=20%`
with capitalization weights `60%`, `25%`, and `15%`, producing `NDX_WDI = 10`.
The suite also covers normalization, missing price/share data, distribution sums,
component contributions, SQLite persistence, provider validation, and all API
routes.
