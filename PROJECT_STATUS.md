# SentinelX â€” Project Status

Last updated: 22 October 2026

## 1. Project Overview

SentinelX is a predictive-maintenance project using NASA C-MAPSS FD001 turbofan engine data.

Main components:
- Remaining Useful Life (RUL) prediction using LightGBM.
- FastAPI prediction and validation endpoints.
- SHAP-based prediction explanations.
- Isolation Forest anomaly detection.
- Anomaly evaluation, sensor comparisons, and alert-persistence analysis.

## 2. RUL Prediction

Model: LightGBM

Previously recorded metrics:
- RMSE: 26.08 cycles
- MAE: 18.95 cycles

These metrics should be reconfirmed against the final model and evaluation outputs before publication.

## 3. Anomaly Detector

Pipeline:
- StandardScaler
- IsolationForest
- 24 input features: 3 operating settings and 21 sensors

Dataset: FD001

Engine-level partition:
- 60% training
- 20% validation
- 20% reserved final test

Split seeds:
- First split: 42
- Second split: 43

Saved model:
models/sentinelx_anomaly_detector.joblib

Frozen threshold:
-0.0692088910956446

Threshold selection:
- 2.5th percentile of validation decision-function scores
- Flag rule: score <= threshold
- Lower scores indicate more unusual readings

Validation flag rate: approximately 2.51%

## 4. Final-Test Evaluation

Reserved final-test engines: 20
Final-test readings: 4,010
Flagged readings: 114
Final-test flag rate: 2.84%
Engines with at least one flagged reading: 20

The detector and threshold were not recalibrated using final-test data.

### Cycle-stage findings

- Cycles 1â€“50: 0.00% flagged
- Cycles 51â€“100: 0.00% flagged
- Cycles 101â€“150: 1.22% flagged
- Cycles 151â€“200: 6.81% flagged
- Cycles 201+: 16.00% flagged

### RUL-stage findings

- RUL 0â€“30: 114 of 620 readings flagged (18.39%)
- RUL 31â€“60: 0 of 600 flagged
- RUL 61â€“100: 0 of 800 flagged
- RUL 101+: 0 of 1,990 flagged

All 114 flagged readings had RUL <= 30 in this final-test partition.

These are descriptive associations, not proof of predictive fault detection.

## 5. Sensor Comparison

Script:
src/analyze_flagged_sensors.py

Output:
reports/flagged_sensor_comparison_by_rul.csv

Within the RUL 0â€“30 group, the largest absolute standardized mean differences were:

- sensor_4: +1.2811
- sensor_15: +1.1619
- sensor_17: +1.1288
- sensor_12: -1.1239
- sensor_20: -1.1120

These are group-level descriptive differences. They are not causal explanations or per-reading feature attributions.

## 6. Alert Persistence

Script:
src/analyze_anomaly_persistence.py

Outputs:
- reports/anomaly_consecutive_runs.csv
- reports/anomaly_persistence_by_engine.csv

Results:
- 114 flagged readings
- 55 consecutive alert runs
- 32 runs of one cycle
- 23 runs lasting at least two consecutive cycles
- Longest run: 8 cycles, engine 73, cycles 206â€“213

Other long runs:
- Engine 17: cycles 271â€“276, 6 cycles
- Engine 97: cycles 197â€“202, 6 cycles
- Engine 56: cycles 270â€“275, 6 cycles

Engine 73 had the highest overall flag rate: 6.10%.

Alert runs describe persistence, not confirmed faults or predictive lead time.

## 7. API and Testing

Implemented endpoints:
- GET /
- GET /health
- POST /predict
- POST /validate
- POST /explain
- POST /anomaly

Latest automated test result:
14 passed, 1 warning in 7.57 seconds.

Warning:
Starlette reports a deprecation warning concerning httpx and TestClient.
This did not cause the tests to fail.

## 8. Important Limitations

- No ground-truth anomaly labels are available.
- Anomaly flags indicate unusual readings, not confirmed engine faults.
- Precision, recall, false-positive rate, and fault-detection accuracy cannot currently be established.
- RUL and cycle-stage associations are descriptive and do not establish causality.
- The threshold must not be tuned against the reserved final-test results.
- Sensor mean differences do not establish which features caused individual flags.

## 9. Next Steps

1. Investigate sensor behavior across the longest flagged runs, especially engine 73.
2. Inspect existing model-comparison and evaluation outputs.
3. Rerun the full automated tests after any code changes.
4. Verify the final API and Docker workflow.
5. Update the README with reproducible setup, training, evaluation, and API instructions.
6. Reconfirm final RUL metrics and document all evaluation limitations.
7. Review dependencies and resolve the TestClient deprecation warning appropriately.

## 10. Reproducibility Rule

Use the saved detector, saved engine partitions, and frozen threshold for final-test analysis. Do not retrain or recalibrate the detector on the final-test partition.
