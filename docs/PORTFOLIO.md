# Portfolio Notes

## Resume Wording

- Developed a Python intraday momentum research pipeline for SPY/QQQ minute data,
  combining lagged volatility bands, VWAP exits, leverage limits, and transaction
  cost accounting with trade and equity exports.
- Added reproducible synthetic-data demos, volume-based capacity diagnostics,
  and automated regression tests for signal history and P&L reconciliation.

Use these as project bullets, and only claim a firm affiliation if that is accurate.
Development used AI coding assistance. Describe the parts you understand and can
defend; do not claim live trading results, institutional usage, or an independently
validated investment strategy.

## Interview Walkthrough

1. Run `python demo.py` to demonstrate the complete workflow without API credentials.
2. Explain why rolling signal estimates are shifted before aggregation.
3. Reconcile initial capital plus net trade P&L with final reported equity.
4. Discuss same-bar signal/execution assumptions and why next-bar fills would
   improve realism.
5. Explain why volume participation is only an initial capacity diagnostic.
6. Describe the need for exchange-calendar validation, out-of-sample evaluation,
   financing costs, and more realistic execution before further research use.
