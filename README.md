# Driving Maneuver Classification: Supervised, Semi-Supervised & Unsupervised Approaches

Detecting aggressive driving maneuvers (harsh braking, acceleration, turns, and lane changes) from raw
smartphone accelerometer and gyroscope signals, using LSTM-based deep learning. The repository compares
three learning paradigms on the same task: a fully supervised LSTM classifier, an LSTM autoencoder for
unsupervised representation learning, and a semi-supervised pipeline that fine-tunes a classifier on top
of the autoencoder's learned features.

## Overview

Inertial Measurement Unit (IMU) sensors — accelerometer and gyroscope — are a low-cost way to detect
unsafe driving behavior without any external hardware. This project frames maneuver detection as a
multivariate time-series classification problem: given a short window of tri-axial motion signals, predict
which maneuver (if any) the driver is performing.

Three complementary approaches are implemented, each in its own notebook:

| Approach | Notebook | Labeled data required | Core idea |
|---|---|---|---|
| **Supervised** | `k_fold_supervised_LSTM.ipynb` | Fully labeled | Stacked LSTM trained end-to-end on hand-engineered sensor features |
| **Unsupervised** | `k_fold_unsupervised_(300)_(mendaly_data).ipynb` | None | LSTM autoencoder learns a compressed latent representation of the sensor stream by reconstruction |
| **Semi-supervised** | `k_fold_semi_supervised_(300, 1000)_(mendaly_data)_final.ipynb` | Partially labeled | The pretrained autoencoder's bottleneck encoder extracts features, which a lightweight classifier is then trained on using only the labeled subset |

## Repository structure

```
.
├── k_fold_supervised_LSTM.ipynb                                  # Supervised LSTM classifier (7 classes, custom sensor dataset)
├── k_fold_unsupervised_(300)_(mendaly_data).ipynb                 # LSTM autoencoder pretraining (Mendeley dataset)
├── k_fold_semi_supervised_(300, 1000)_(mendaly_data)_final.ipynb  # Classifier on top of pretrained encoder features
└── README.md
```

> **Note:** the raw sensor CSVs and trained model weights (`.h5`) are not included in this repository. Each
> notebook currently reads from and saves to hardcoded local paths (e.g. `D:/data/...`) — update these paths
> to your own data/output locations before running.

## Datasets

Two different IMU datasets are used across the notebooks:

1. **Custom driving-session dataset** (`k_fold_supervised_LSTM.ipynb`) — four recorded sessions
   (`acelerometro_terra_{16,17,20,21}.csv`, `giroscopio_terra_{16,17,20,21}.csv`,
   `groundTruth_{16,17,20,21}.csv`), each providing accelerometer/gyroscope streams plus a ground-truth
   event log. Seven event classes: `non_aggressive`, `aggressive_acceleration`, `aggressive_braking`,
   `aggressive_left_turn`, `aggressive_right_turn`, `aggressive_left_lane_change`,
   `aggressive_right_lane_change`.
2. **Mendeley Data driving dataset** (`raw.csv`, used by the unsupervised and semi-supervised notebooks) —
   tri-axial accelerometer/gyroscope readings with a `Target` label (1–4), mapped to
   `aggressive_acceleration`, `aggressive_right_turn`, `aggressive_left_turn`, and `aggressive_braking`.
   *(Add the dataset's DOI/citation here once confirmed — the notebooks reference it only as
   `Mendeley Data/raw.csv`.)*

Neither dataset is redistributed in this repository; both are read from local paths set at the top of each
notebook's data-loading cell.

## Methodology

### Feature engineering

Both pipelines derive features from the raw `acc_x`, `acc_y`, `gyro_z` streams using a shared recipe:

- **Energy terms:** squared signal (`eng_acc_x`, `eng_acc_y`, `eng_gyro_z`)
- **Slope/derivative:** first difference over time (`slope_acc_x`, `slope_acc_y`, `slope_gyro_z`)
- **Rolling-window statistics** (window = 20 samples, shifted by 1 to avoid leakage): mean, min, max,
  variance, and standard deviation per axis
- **Rule-based aggressive-event flags** (`agg_ACC`, `agg_Brake`, `agg_LT`, `agg_RT`, `agg_LLC`, `agg_RLC`),
  derived from fixed thresholds on acceleration/gyro magnitude and slope, used alongside the learned
  features

Labels are one-hot encoded (`LabelEncoder` + `to_categorical`) and inputs are scaled
(`StandardScaler`/`MinMaxScaler`) before being reshaped to `(samples, timesteps, 1)` for the LSTM input.

### Model architectures

**Supervised classifier** — a stacked LSTM trained end-to-end:

```python
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_features, 1)))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

**Unsupervised autoencoder** — a symmetric LSTM encoder/decoder with a 10-unit bottleneck, trained to
reconstruct its own input (`X`, `X`):

```python
encoder_decoder = Sequential()
encoder_decoder.add(LSTM(31, activation='elu', return_sequences=True, input_shape=(n_features, 1)))
encoder_decoder.add(Dropout(0.3))
encoder_decoder.add(LSTM(20, activation='elu', return_sequences=True))
encoder_decoder.add(Dropout(0.3))
encoder_decoder.add(LSTM(10, activation='elu', return_sequences=False, name='bottleneck'))
encoder_decoder.add(RepeatVector(31))
encoder_decoder.add(LSTM(10, activation='elu', return_sequences=True))
encoder_decoder.add(Dropout(0.3))
encoder_decoder.add(LSTM(20, activation='elu', return_sequences=True))
encoder_decoder.add(Dropout(0.3))
encoder_decoder.add(LSTM(31, activation='elu', return_sequences=True))
encoder_decoder.add(TimeDistributed(Dense(1)))
encoder_decoder.compile(loss='mse', optimizer='adam')
```

The `bottleneck` layer is exported as a standalone encoder model and reused by the semi-supervised
notebook to produce a compact latent representation of the labeled data, which a classifier head is then
trained on.

### Evaluation protocol

All three notebooks use `k`-fold cross-validation (`k=10`, `random_state=42`) — `StratifiedKFold` for the
classification tasks (preserves class balance per fold) and plain `KFold` for the unsupervised
reconstruction task. Classification is scored with accuracy, macro-averaged precision/recall/F1, ROC-AUC
(one-vs-rest per class), and a confusion matrix; the autoencoder is scored with reconstruction MSE/RMSE.

## Results

Metrics below are taken directly from each notebook's saved execution output (10-fold cross-validation).

**Supervised LSTM** (7-class, custom driving-session dataset):

| Fold | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 1 | 0.9666 | 0.9599 | 0.9591 | 0.9595 |
| 2 | 0.9404 | 0.9310 | 0.9262 | 0.9268 |
| 3 | 0.6832 | 0.7438 | 0.5797 | 0.5989 |
| 4 | 0.9025 | 0.9129 | 0.8388 | 0.8653 |
| 5 | 0.6877 | 0.6724 | 0.5850 | 0.5970 |
| 6 | 0.9404 | 0.9327 | 0.9405 | 0.9355 |
| 7 | 0.9603 | 0.9585 | 0.9504 | 0.9542 |
| 8 | 0.9557 | 0.9500 | 0.9490 | 0.9492 |
| 9 | 0.9250 | 0.9162 | 0.9213 | 0.9184 |
| 10 | 0.9783 | 0.9777 | 0.9780 | 0.9777 |

**Overall (aggregated across held-out folds):** Accuracy **0.8940**, Recall **0.8624**, F1 **0.8778**.
Folds 3 and 5 are notable outliers (~0.68 accuracy vs. ~0.90–0.98 elsewhere) — worth investigating (e.g.
class imbalance in that particular fold split) before treating the aggregate as representative.

**Semi-supervised classifier** (4-class, Mendeley dataset, trained on pretrained-encoder features):

| Fold | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 1 | 0.9821 | 0.9865 | 0.9795 | 0.9826 |
| 2 | 0.9643 | 0.9703 | 0.9629 | 0.9658 |
| 3 | 0.9554 | 0.9571 | 0.9547 | 0.9548 |
| 4 | 0.9107 | 0.9184 | 0.8990 | 0.9029 |
| 5 | 0.9730 | 0.9713 | 0.9686 | 0.9698 |
| 6 | 0.9550 | 0.9528 | 0.9586 | 0.9549 |
| 7 | 0.9459 | 0.9451 | 0.9513 | 0.9478 |
| 8 | 0.9189 | 0.9138 | 0.9157 | 0.9135 |
| 9 | 0.9459 | 0.9443 | 0.9443 | 0.9443 |
| 10 | 0.9279 | 0.9325 | 0.9358 | 0.9314 |

**Mean accuracy across folds: 0.9479** *(computed from the fold-level scores above; the notebook does not
print an aggregate line for this run).*

**Unsupervised autoencoder** (reconstruction quality, Mendeley dataset):

| Fold | RMSE |
|---|---|
| 1 | 18.121 |
| 2 | 16.586 |
| 3 | 14.316 |
| 4 | 16.377 |
| 5 | 17.547 |
| 6 | 16.809 |
| 7 | 13.105 |
| 8 | 13.241 |
| 9 | 16.957 |
| 10 | 9.445 |

**Overall RMSE: 15.466**

## Getting started

### Requirements

No `requirements.txt` is currently checked in. The notebooks were authored against an older Keras/TensorFlow
API surface (standalone `keras` imports, `keras.wrappers.scikit_learn`, `scipy.interp`) — pin compatible
versions rather than installing the latest releases:

```bash
pip install tensorflow==2.3.* "scipy<1.8" scikit-learn pandas numpy matplotlib seaborn
```

`plot_model` (used optionally for architecture diagrams) additionally requires `pydot` and Graphviz.

### Running the notebooks

1. Clone the repository and install the dependencies above.
2. Update each notebook's hardcoded data paths (input CSVs) and model-output paths (currently
   `D:/...` Windows paths) to match your environment.
3. Run the notebooks in this order, since the semi-supervised notebook depends on an encoder model
   produced by the unsupervised notebook:
   1. `k_fold_unsupervised_(300)_(mendaly_data).ipynb` — trains and saves the autoencoder + bottleneck
      encoder per fold.
   2. `k_fold_semi_supervised_(300, 1000)_(mendaly_data)_final.ipynb` — loads a saved encoder, extracts
      features, and trains the classifier head.
   3. `k_fold_supervised_LSTM.ipynb` — fully independent; can be run on its own against the custom
      driving-session dataset.

## Known limitations

- Data loading paths and saved-model paths are hardcoded per-machine (`D:/data/...`) rather than
  configurable via arguments or a config file.
- No `requirements.txt`, environment file, or license is currently included.
- The rule-based aggressive-event flags (fixed acceleration/gyro thresholds) are hardcoded in-notebook
  rather than exposed as tunable parameters.
- Supervised-model fold variance (folds 3 and 5, see Results) suggests the current 10-fold split isn't
  fully robust to class imbalance across folds.

## License

No license file is currently included in this repository. Add one (e.g., MIT, Apache-2.0) to clarify how
others may use or build on this work.
