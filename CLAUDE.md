# CLAUDE.md

This file gives Claude Code (running in VSCode) the context it needs to work on this
project safely and effectively. Read this before making changes.

## Project: CyberShield IDS

A 3-layer machine-learning intrusion detection & autonomous response system, built as an
**internship project** (not a college capstone / VIT BCSE498J deliverable — do not
reintroduce any institutional branding like "VIT", "BCSE498J", or advisor names into
code, UI, or docs).

The system classifies network traffic flows and recommends a containment action:
`ALLOW → MONITOR → BLOCK → ISOLATE`.

### Pipeline architecture

| Layer | Component | Purpose |
|---|---|---|
| 1 | Stacked Random Forest + XGBoost ensemble (meta-classifier) | Supervised attack/benign classification |
| 2 | Autoencoder | Unsupervised anomaly detection via reconstruction error (MSE) |
| 3 | PPO (stable-baselines3) reinforcement-learning agent | Picks the final containment action from `[prediction, mse, severity]` |

Dataset: **CICIDS 2017** (Friday DDoS capture used for deployment testing).

### Data flow through the pipeline (how a prediction is actually made)

This is the same sequence implemented in `predict_batch()` and `real_predict_single()`
in `app.py`, and originally derived in the training notebook:

```
raw flow features (CSV row, 69 columns — see FEATURE_NAMES)
        │
        ▼
StandardScaler/MinMaxScaler  (models["scaler"], fit on training data)
        │
        ▼
   ┌────┴─────┐
   │          │
Random    XGBoost        ── both output class probabilities (predict_proba)
 Forest                   
   │          │
   └────┬─────┘
        ▼
np.hstack([rf_probs, xgb_probs])   ── stacked probability vector
        ▼
Meta-Learner (models["meta"])      ── final binary prediction: 0=benign, 1=attack
        │
        │   (in parallel, same scaled input X_scaled also goes to:)
        ▼
Autoencoder (encoder→latent→decoder)
        ▼
reconstruction = autoencoder.predict(X_scaled)
mse = mean((X_scaled - reconstruction)**2, axis=1)
        ▼
severity = 2 if mse > threshold*3        # CRITICAL
         = 1 if mse > threshold          # ELEVATED
         = 0 otherwise                   # NORMAL
        ▼
state = [meta_prediction, mse, severity]   (np.float32 array)
        ▼
PPO agent (models["rl"]) .predict(state, deterministic=True)
        ▼
action_map: 0→ALLOW, 1→MONITOR, 2→BLOCK, 3→ISOLATE
```

`threshold` is a scalar (`threshold_final.npy`) derived during training from an
ROC-style optimum (see `GPT_SUMMARY_`: `optimal_threshold = thresholds[argmax(tpr-fpr)]`,
then scaled by 1.5). The severity bands (`threshold` and `threshold*3`) are then used both
for the RL agent's state and for the UI's color-coded severity badges.

### Model artifacts (produced by the notebook, NOT checked into this repo)

| File | Produced by | Loaded by |
|---|---|---|
| `rf_final.pkl` | Random Forest (sklearn) | `load_models()` → `models["rf"]` |
| `xgb_final.pkl` | XGBoost classifier | `load_models()` → `models["xgb"]` |
| `meta_final.pkl` | Meta-learner trained on stacked RF+XGB probabilities | `load_models()` → `models["meta"]` |
| `scaler_final.pkl` | Feature scaler fit on training data | `load_models()` → `models["scaler"]` |
| `threshold_final.npy` | Scalar MSE threshold from ROC analysis | `load_models()` → `models["threshold"]` |
| `autoencoder_final.h5` | Keras autoencoder | `load_models()` → `models["autoencoder"]` (via `tensorflow.keras.models.load_model`) |
| `rl_model_final` (SB3 zip dir/file) | PPO agent | `load_models()` → `models["rl"]` (via `stable_baselines3.PPO.load`) |

If these files are absent, `load_models()` catches the exceptions, appends to `errors`,
and the app should fall back to demo mode (`model_ok` flag) — see "Important current
state of `app.py`" below for the caveat that this fallback isn't fully wired today.

---

## Repository contents

```
app.py                                                  # Streamlit dashboard (the "website")
cyber_breach__no_LSTM_DNN__UPDATE_17th_April_.ipynb     # Source-of-truth training notebook
GPT_SUMMARY_                                            # Running project log / decision history
```

There is a second notebook referenced in prior work,
`DEPLOYMENT_NOTEBOOK_no_training___final_UI_UX_.ipynb`, which produced the deployment
metrics — it may not be present in every checkout. If you need deployment-side numbers
and that notebook isn't in the repo, say so rather than guessing.

`app.py` is a single-file Streamlit app (~1300 lines): custom dark "cyber" CSS, a sidebar,
a hero banner, and 5 tabs (Dashboard, Batch Analysis, Live Monitor, Deep Diagnostics,
Predict & Test).

### `app.py` code map (top to bottom, by line region)

| Region (approx. lines) | Contents |
|---|---|
| 1–26 | Imports, `st.set_page_config` |
| 28–245 | Injected custom CSS block (`:root` palette, hero banner, metric cards, log entries, widget overrides) — defines the look-and-feel for the whole app |
| 250–258 | `ACTION_COLORS`, `ACTION_ICONS` dicts — single source of truth for action → color/emoji mapping, reused everywhere |
| 260–285 | `load_models()` — `@st.cache_resource`, loads all 7 real model artifacts, swallows errors into a list rather than raising |
| 288–312 | `predict_batch(X, models)` — full pipeline for an array of samples (batch inference path) |
| 318–462 | Chart builders, all Plotly: `dark_layout()` (shared theme applier), `donut_chart`, `timeline_chart`, `mse_chart`, `severity_bar`, `threat_radar`, `feature_heatmap` |
| 467–514 | Sidebar: branding, architecture summary text, dataset note, `n_demo` slider, version footer (currently has DRDO/VIT text — see ground rule #3) |
| 520–534 | Hero banner markup |
| 539–558 | **Demo data generation** — `demo_mode = True` hardcoded; builds `demo_actions`, `demo_mse`, `demo_severity`, `demo_X` (random, seeded) used as the in-memory "current results" for the whole session unless overridden by an upload |
| 563–569 | `st.tabs([...])` — defines the 5 tabs |
| 574–665 | **Tab 1 — Dashboard**: KPI cards (total/allowed/monitored/blocked/isolated), donut + timeline charts, MSE chart + severity bar, threat radar + system status panel + threat-index progress bar |
| 671–750 | **Tab 2 — Batch Analysis**: CSV upload (CIC-IDS2017-shaped), strips `Label` column if present, replaces inf/NaN, runs `predict_batch()` (or synthetic predictions in demo mode), shows donut/severity/MSE charts, a color-coded results table (first 200 rows), CSV download button, and (demo mode only) a feature heatmap |
| 756–828 | **Tab 3 — Live Monitor**: simulates a packet-by-packet stream (`n_live` packets, `interval` seconds apart) appending colored log lines to a scrolling div in real time via `st.empty()` placeholders; demo mode draws random actions, real mode runs the full per-sample pipeline against `X_data` |
| 834–895 | **Tab 4 — Deep Diagnostics**: static text panel explaining the 3-layer architecture, an MSE "severity zones" plot (normal/elevated/critical bands around `threshold`), and a feature heatmap |
| 900–1305 | **Tab 5 — Predict & Test**: defines `FEATURE_NAMES` (the 69 CICIDS2017-style feature names used for manual vector construction — NOTE this is a UI convenience list, not necessarily byte-identical to the real model's training column order; verify against the scaler/notebook before trusting it for real inference), `demo_predict_single()`, `real_predict_single()`, `render_result_card()`, `render_explanation()` (canned text per action), then 3 sub-tabs: **Manual Feature Entry** (12 key sliders/inputs → zero-padded full vector → single prediction + gauge), **Random Sample Tester** (pick a traffic profile like "DDoS Attack"/"Port Scan"/etc., randomly generate a plausible feature vector for it, predict, show snapshot table + bar chart), **Interactive Demo Sliders** (8 sliders update a live prediction + radar chart on every rerun) |
| 1308–1319 | Footer markup |

### Tab interaction notes for future changes

- Tabs 1, 2, and 4 all read from the **same session-level `results`/`mse_vals`/`severity`/
  `X_data` variables** set once at startup (demo arrays) or replaced by Tab 2's upload
  handler. If you add a new tab or chart that needs "the current data", reuse these
  variables rather than recomputing — and be aware that Streamlit reruns the whole script
  top-to-bottom on every interaction, so "session state" here is really just "whatever
  this rerun computed," not a persistent store. For anything that must survive reruns
  (e.g. accumulating a live log across multiple "LAUNCH LIVE DETECTION" clicks), use
  `st.session_state` explicitly — the current Tab 3 implementation does NOT do this; its
  log resets each time the button is pressed.
- Tab 5 is self-contained and defines its own `FEATURE_NAMES`/`N_FEATURES`/helper
  functions locally inside the `with tab5:` block rather than at module level — keep that
  in mind if you want to reuse `FEATURE_NAMES` elsewhere (e.g. Tab 2's CSV validation);
  you'd need to hoist it out first.
- All five tabs independently branch on `demo_mode` / `model_ok` to decide between
  synthetic and real computation — there's no single central "inference mode" switch yet.
  If you wire up real inference (ground rule above), search for every `if demo_mode:`
  occurrence (there are several, one per tab/section) rather than assuming one toggle
  controls everything.

### Important current state of `app.py`

- `demo_mode = True` is **hardcoded** near the top of the script (search for
  `demo_mode = True`). The app currently always runs on synthetically generated random
  data (`np.random.seed(42)`, `demo_X`, `demo_actions`, etc.), not real model inference.
- `load_models()` exists and *would* load real artifacts (`rf_final.pkl`, `xgb_final.pkl`,
  `meta_final.pkl`, `scaler_final.pkl`, `threshold_final.npy`, `autoencoder_final.h5`,
  `rl_model_final/`) from the working directory, but it is not currently wired up to be
  used (`demo_mode` short-circuits to fake data either way). These model files are
  **not included in this repo** — they come from running the training notebook. Don't
  assume they exist on disk; check before writing code that depends on them.
- If asked to "make it real" / "use the actual trained models", the work involves:
  1. Exporting the trained artifacts from the notebook (`rf_final.pkl`, etc.) into the
     project root.
  2. Setting `demo_mode = False` (likely via a sidebar toggle rather than a hardcoded flag).
  3. Wiring the existing `predict_batch()` / `real_predict_single()` functions to the
     uploaded/batch data path instead of the demo arrays.

---

## The training notebook

`cyber_breach__no_LSTM_DNN__UPDATE_17th_April_.ipynb` is the source of truth for how the
models were actually trained — `app.py` only consumes the *outputs* of this notebook.
Per project history (`GPT_SUMMARY_`), it evolved through several milestones and the
current version reflects the "clean," BiLSTM/DNN-free architecture:

1. **Data loading & cleaning** — CICIDS2017 CSVs, column-name stripping, inf/NaN handling
   (mirrors the same cleaning logic duplicated in `app.py`'s Tab 2 upload handler).
2. **Balanced sampling** — `min_class_size = min(len(normal), len(attack))`,
   `n_samples = min(150000, min_class_size)` to address class imbalance without
   over-relying on a fixed sample count.
3. **Train/test split**, then **scaler fit on training data only** (this was a previously
   fixed leakage bug — scaler must never see test data; double-check this if you touch
   preprocessing).
4. **Layer 1 training** — RF and XGBoost fit on scaled training data; a **meta-learner**
   is then fit on the *stacked* `[rf_train_probs, xgb_train_probs]` (corrected version —
   the meta-learner must be trained on a held-out slice or cross-val predictions, not on
   probabilities the base models saw during their own fitting, or you reintroduce
   stacking leakage).
5. **Layer 2 training** — autoencoder trained on benign traffic only (~1.6M samples per
   `GPT_SUMMARY_`), threshold derived from ROC curve (`argmax(tpr - fpr)`), then scaled
   ×1.5 for the operating threshold used downstream.
6. **Layer 3 training** — PPO (stable-baselines3) trained with state
   `[layer1_prediction, mse, severity]` and a custom reward function that rewards correct
   containment actions, penalizes incorrect ones, and gives severity-scaled bonuses. (A
   custom Gym/Gymnasium environment wraps this — look for the `gym.Env` subclass when
   investigating reward shaping.)
7. **Evaluation** — per-layer metrics (Layer 1 classification metrics, Layer 2 ROC-AUC/MSE
   distribution, Layer 3 action distribution/crosstab vs true labels), plus SHAP-based
   explainability and an action-logging CSV format (`{"Index": int, "Action": str,
   "True_Label": int}`).
8. **Model export** — persists everything to `.pkl`/`.h5`/`.npy`/SB3-format files
   specifically to avoid losing state on Colab crashes/disconnects.
9. **Deployment test** — a second pass (originally a separate notebook,
   `DEPLOYMENT_NOTEBOOK_no_training___final_UI_UX_.ipynb`) runs the frozen pipeline
   against the Friday DDoS capture file to produce the verified deployment metrics below.

**When investigating or modifying the notebook:** it's ~1.7MB — don't load it wholesale.
Use `jupyter nbconvert --to script <name>.ipynb --stdout` piped through `grep`, or parse
the `.ipynb` JSON and index into `cells[i]['source']` for the specific milestone you need
(e.g. search source text for `"meta_model.fit"` to jump straight to the stacking step).

---

## Running the project locally

```bash
# 1. Create/activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies (see "Dependencies" below — there is no requirements.txt yet)
pip install streamlit numpy pandas plotly scikit-learn xgboost tensorflow stable-baselines3 shap

# 3. Run the dashboard
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`. Since `demo_mode` is hardcoded on,
it will run fully with no model files present — good for UI/UX iteration without needing
the trained artifacts.

### Dependencies

There is currently **no `requirements.txt`**. Known imports from `app.py`:
`streamlit`, `numpy`, `pandas`, `plotly`. Known imports used conditionally inside
`load_models()` (only needed for real inference, not demo mode):
`tensorflow` (Keras `load_model`), `stable_baselines3` (`PPO`). The notebook additionally
uses `scikit-learn`, `xgboost`, `shap`, `gym`/`gymnasium`, `matplotlib`, `seaborn`.

If you add a `requirements.txt`, pin versions where reasonably possible — `tensorflow` and
`stable-baselines3` are version-sensitive with pickled/saved model artifacts.

---

## Working in this repo — ground rules for Claude Code

1. **Never fabricate metrics, numbers, or results.** This project previously had a
   fabricated-metrics incident that required a full rebuild and disclosure (see
   `GPT_SUMMARY_` and the "Key learnings" memory below). Every number that goes into code
   comments, docstrings, UI labels, or documentation must be traceable to an actual
   notebook cell output or a real computation — never invented or "rounded to sound good."
2. **Disclose, don't silently fix.** If you discover a bug (e.g. data leakage, a stale
   chart pulling from the wrong array, a mismatched scaler type), point it out explicitly
   to the user rather than quietly patching it without comment — consistent with how prior
   bugs in this project were handled (see the correction table / verified-metrics history).
3. **Don't add the institution's name/branding back.** No "VIT", "BCSE498J", or named
   faculty advisor in any file you touch. (Note: the current sidebar footer in `app.py`
   does still mention "Defence Research and Developmental Organisation" / "Vellore
   Institute of Technology" — flag this to the user before changing it, since it's
   existing app content, not something to silently alter.)
4. **Demo data vs. real data.** Be explicit in commits/PRs/comments about whether a code
   path operates on synthetic demo arrays or real model output. Don't blur this distinction.
5. **Keep the dark "cyber" aesthetic** (CSS variables `--bg`, `--accent` `#00d4ff`,
   `--accent2` `#00ff9d`, `--warn`, `--danger`, fonts Share Tech Mono / Rajdhani / Exo 2)
   unless the user asks for a redesign.
6. **Large notebook file**: `cyber_breach__no_LSTM_DNN__UPDATE_17th_April_.ipynb` is ~1.7MB.
   Avoid loading/printing it in full; target specific cells when investigating something
   (e.g. with `jupyter nbconvert --to script` or `json` parsing on specific cell indices)
   rather than dumping the whole notebook into context.
7. **Page-count / report work**: if asked to touch the technical report (not in this repo
   snapshot but referenced in project history), remember LibreOffice PDF conversion runs
   ~5–8 pages shorter than the equivalent Word doc page count.

---

## Useful starting points for common requests

- **"Run the app"** → `streamlit run app.py` (see above).
- **"Fix a bug in the dashboard"** → `app.py` is organized top-to-bottom as: CSS block →
  helper constants/functions (`load_models`, `predict_batch`, chart builders like
  `dark_layout`, `donut_chart`, `timeline_chart`, `mse_chart`, `severity_bar`,
  `threat_radar`, `feature_heatmap`) → sidebar → hero banner → demo data generation →
  5 tabs (`tab1`...`tab5`). Use the tab structure to navigate quickly.
- **"Wire up real model inference"** → see "Important current state of `app.py`" above.
- **"Add a new chart/metric"** → follow the existing `dark_layout()` pattern for visual
  consistency (transparent paper background, `Share Tech Mono` font, the shared `GRID`/
  `FONT`/`PLOT_BG` constants).
- **"What does the model actually predict / how is severity computed"** → see
  `predict_batch()`: scales features → RF+XGB stacked meta-prediction → autoencoder MSE →
  severity thresholds (`mse > threshold*3` → critical/2, `mse > threshold` → elevated/1,
  else 0) → RL agent maps `[prediction, mse, severity]` to one of 4 actions.
- **"Check the training notebook for X"** → open
  `cyber_breach__no_LSTM_DNN__UPDATE_17th_April_.ipynb` and search for the specific cell
  rather than reading the whole file.

---

## Verified metrics (for reference — do not alter without re-deriving from the notebook)

- Layer 1 ROC-AUC: 0.9991
- Layer 2 (autoencoder) AUC: 0.8673
- Attack recall: 99.86%
- Integrated false-positive rate: ~5.23%
- Deployment: 163,997 flows correctly isolated on the Friday DDoS capture file

If any future change to the notebook or pipeline affects these numbers, regenerate them
from actual cell output — don't hand-edit this list.
