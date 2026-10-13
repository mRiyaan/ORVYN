# Phase 12 Day-Wise Data Audit

## 1. Source Files Analyzed

| Source File | Timestamp Type | Recording Duration | Calendar-Day Availability | Reproducibility Status |
| :--- | :--- | :--- | :--- | :--- |
| `outputs/phase9/daily_outputs.csv` | Calendar Date | 7 fabricated days | YES (but synthetic) | CANNOT be reproduced from WESAD source data. |
| `outputs/demo/person_01/daily_outputs.csv` | Relative Seconds (`start_time_s`) | ~101 minutes | NO | YES |
| `outputs/demo/person_02/daily_outputs.csv` | Relative Seconds (`start_time_s`) | ~87 minutes | NO | YES |

## 2. Findings
1. **Which files contain real calendar timestamps?** None of the original WESAD-derived files. `outputs/phase9/daily_outputs.csv` contains fabricated calendar dates.
2. **Which files contain only relative seconds?** Both `person_01/daily_outputs.csv` and `person_02/daily_outputs.csv` contain only relative seconds (`start_time_s`).
3. **Which files actually span more than one calendar day?** None of the real WESAD demo files span more than one day. The fabricated Phase 9 file spans 7 days.
4. **Which daily values can be reproduced from source data?** None of the daily values in the fabricated Phase 9 file can be reproduced from WESAD source data.
5. **Whether the existing Phase 12 report is consistent with the actual daily artifacts?** The existing Phase 12 report is inconsistent because it claims a 2026-10-01 to 2026-10-07 timeline, whereas the actual data is a ~101 minute continuous recording without calendar dates.

## 3. Action Plan
- Remove or explicitly ignore the fabricated Phase 9 daily outputs.
- In the dashboard, detect if the data spans multiple calendar days. If not (like WESAD demo data), display a "Session-Level Summary" fallback.
- For `daywise_reconciliation.csv`, since the fabricated Phase 9 data cannot be reconciled with source data, it will be marked as such.
