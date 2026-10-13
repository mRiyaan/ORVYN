# Phase 1: RELAX Source Audit

## Dataset Source
* Dataset: **RELAX: A six-week longitudinal dataset of wearable and self-reported stress measurements in working adults**
* DOI: [10.5281/zenodo.20701999](https://doi.org/10.5281/zenodo.20701999)
* Source Zip: `RELAXDataset.zip` (~16.5 GB)

## Selected Participant
**Participant ID:** `57`

Participant 57 was selected because they provide an extensive valid longitudinal coverage of 61 calendar days with a significant number of continuous physiological measurements (IBI) and overlapping psychological context (self-reports).

## Audit Metrics
* **First Timestamp:** `2024-02-26 08:48:03 UTC`
* **Last Timestamp:** `2024-04-27 20:00:40 UTC`
* **Calendar Days Span:** 61 days
* **IBI Data Rows:** 1,315,568 rows
* **IBI Frequency:** Event-based (not uniform continuous sampling)
* **ACC Data:** Skipped/Missing locally due to size constraints.
* **Missingness:** Since ACC is unavailable, the orchestrator will treat ACC fields as missing and drop those dependent features cleanly without fabricating data.
* **Self-Reports:**
  * Morning Questionnaire (`mfb`): 37 entries
  * Intra-day Questionnaire (`ifb`): 72 entries
  * Evening Questionnaire (`afb`): 28 entries

## Authentication and Immutability
All timestamps are authentic. No physiological values were altered, and no synthetic BVP, EDA, or TEMP signals were created. The data remains separated from the WESAD pipeline logic as per the overarching architectural rules.
