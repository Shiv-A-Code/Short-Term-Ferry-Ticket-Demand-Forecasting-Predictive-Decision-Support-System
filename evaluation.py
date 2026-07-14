import numpy as np
import pandas as pd

def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def peak_miss_rate(y_true, y_pred, threshold=0.9):
    peak_threshold = np.quantile(y_true, threshold)
    missed_peaks = ((y_true >= peak_threshold) & (y_pred < peak_threshold)).sum()
    total_peaks = (y_true >= peak_threshold).sum()
    return missed_peaks / total_peaks if total_peaks > 0 else 0

def confidence_band_width(forecast_df):
    if 'yhat_upper' in forecast_df and 'yhat_lower' in forecast_df:
        return (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean()
    return None

def forecast_lead_time(y_true, y_pred, horizon=1):
    spikes = np.where(y_true > np.quantile(y_true, 0.9))[0]
    lead_times = []
    for s in spikes:
        if s-horizon >= 0 and y_pred[s-horizon] > np.quantile(y_true, 0.9):
            lead_times.append(horizon)
    return np.mean(lead_times) if lead_times else 0

def kpi_summary(y_true, y_pred, forecast_df=None, horizon=1):
    metrics = evaluate(y_true, y_pred)
    kpis = {
        "Peak Miss Rate": peak_miss_rate(y_true, y_pred),
        "Forecast Lead Time": forecast_lead_time(y_true, y_pred, horizon),
        "Confidence Band Width": confidence_band_width(forecast_df) if forecast_df is not None else None
    }
    summary = {**metrics, **kpis}
    return pd.DataFrame(summary, index=["Values"]).T

