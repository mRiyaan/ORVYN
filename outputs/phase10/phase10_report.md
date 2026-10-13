# Phase 10 Report — Interactive Demonstration Dashboard

**Date**: 2026-10-07  
**Status**: READY / VERIFIED  
**Application Entrypoint**: [`app/dashboard.py`](../../app/dashboard.py)  

---

## 1. Executive Summary
Phase 10 provides a full-featured, real-time interactive Streamlit web application:
1. **Single-Subject Deep-Dive**: Select any WESAD subject, navigate their experimental timeline window-by-window, and view their personalized resting baseline card.
2. **Real-Time Multimodal Oscilloscope**: Interactive Altair charts rendering synchronous 6.0-second wrist streams:
   - Blood Volume Pulse (BVP @ 64 Hz)
   - Electrodermal Activity (EDA @ 64 Hz)
   - Skin Temperature (TEMP @ 64 Hz)
3. **Personalized Strain Index (0 - 100)**: Real-time radial score and color-coded clinical state.
4. **Physiological Cause Attribution**: Decomposition into sympathetic surge, tachycardia, and vagal withdrawal.
5. **Recovery Trajectory Forecast**: Dynamic projection curve showing expected recovery minutes.
6. **Two-Person Comparative Live Demo**: Side-by-side demonstration comparing two individuals under identical stress.

---

## 2. Launch Instructions
To launch the interactive dashboard locally:
```bash
streamlit run app/dashboard.py
```
