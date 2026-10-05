# AGRIRAIN AI — Rainfall-Based Crop Yield Forecasting Using Gradient Boosting

## Run (VS Code terminal, inside this folder)
```
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000/ (redirects to /overview). First start takes ~25 s while the model trains once; it is then cached.

## Structure
app.py (routes) · data_processing.py (load/validate/stats) · model.py (GB regressor + supplementary classifier) · charts.py (Plotly specs)
templates/ (base + 8 pages) · static/css, static/js · data/crop_yield.csv

## How the model works
Target = Yield. Features = Crop, Season, State (one-hot) + Crop_Year, Area, Production, Annual_Rainfall, Fertilizer, Pesticide.
80/20 split, random_state=42. GradientBoostingRegressor adds 200 depth-4 trees one after another; each tree fits the residual errors of the ensemble so far.
The same cached model serves the Gradient Boosting and Forecast pages. The confusion matrix comes from a separate Low/Medium/High classifier (tertiles) and is labelled supplementary.

## Viva Q&A
1. **Why regression?** Yield is continuous. 
2. **Why Gradient Boosting?** Handles non-linear effects and mixed feature types; sequential trees correct earlier errors.
3. **Why is R² so high?** Yield = Production/Area, and both are inputs (leakage); also a few huge-yield outliers dominate R². The GB page shows a Production-removed check.
4. **Why is MAE more representative than RMSE here?** RMSE is dominated by outlier crops (e.g. coconut).
5. **Why one-hot encoding?** Crop/State/Season are nominal; integer codes would imply a false order.
6. **Why does rainfall correlate weakly with yield?** Pooled across all crops, crop type dominates; effect is crop/region specific.
7. **Why Spearman as well as Pearson?** Rank-based, robust to extreme skew.
8. **Is the confusion matrix evaluating the regressor?** No; separate supplementary classifier.
9. **Can it forecast future years?** No; trees can't extrapolate, so Crop_Year is limited to 1997–2020.
10. **What does feature importance mean?** Total error reduction attributed to a feature across trees — not causation.
