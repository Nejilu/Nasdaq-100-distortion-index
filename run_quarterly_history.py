"""Rebuild the quarterly QQQ-versus-SPGM history from SEC N-PORT filings."""

from __future__ import annotations

import argparse

from edgar_quarterly_history import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_HISTORY_PATH,
    build_quarterly_history,
    save_quarterly_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build quarterly NDX-WDI history from SEC EDGAR filings."
    )
    parser.add_argument("--output", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    history = build_quarterly_history(
        archive_dir=args.archive_dir,
        max_workers=args.max_workers,
    )
    destination = save_quarterly_history(history, args.output)
    print(
        f"Saved {len(history)} quarterly observations "
        f"({history['report_date'].min()} to {history['report_date'].max()}) "
        f"to {destination}."
    )


if __name__ == "__main__":
    main()
