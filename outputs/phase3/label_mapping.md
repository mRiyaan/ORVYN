# Official WESAD Label Mapping & Downstream Semantic Formulation

## 1. Ground Truth Source
Derived directly from the official WESAD publication (Schmidt et al., ICMI 2018) and the accompanying `wesad_readme.pdf`.

## 2. Full Raw Code Definition

| Label Code | Semantic Condition | Study Protocol Description |
|:---:|:---|:---|
| **0** | Undefined / Transient | Transition intervals between study stages, equipment setup, or unannotated periods |
| **1** | Baseline | 20-minute neutral resting period reading magazines on a comfortable chair |
| **2** | Stress | Trier Social Stress Test (TSST): 10 min public speaking preparation + arithmetic challenge under social evaluation |
| **3** | Amusement | Funny video clips watching session (~392 seconds) |
| **4** | Meditation | Guided meditation / relaxation exercise (~420 seconds) |
| **5** | Rest / Questionnaire | Post-task recovery or questionnaire filling interval |
| **6** | Rest / Questionnaire | Post-task recovery or questionnaire filling interval |
| **7** | Rest / Questionnaire | Post-task recovery or questionnaire filling interval |

---

## 3. Downstream Binary Mapping Rule (Phase 4 Handoff)

In strict accordance with the capstone engineering rules:
- **Baseline (`Class 0`)**: Raw Code `1` (Neutral rest)
- **Stress (`Class 1`)**: Raw Code `2` (Trier Social Stress Test)

> [!IMPORTANT]
> **Strict Semantic Boundary**: Transient periods (`0`), amusement (`3`), meditation (`4`), and rest intervals (`5-7`) **MUST NOT** be silently collapsed into "non-stress". Collapsing amused or deeply meditative physiology into the neutral baseline distorts the autonomic signature. They are explicitly isolated and excluded from the primary binary baseline-vs-stress evaluation, matching the primary evaluation protocol of Schmidt et al. (2018).
