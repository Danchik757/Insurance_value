import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.storage import DatabaseStorage

VERSION = "1.0.0"

logger = setup_logger("DataAnalysis", log_file=CONFIG["logging"]["path"], level=CONFIG["logging"]["level"])

class DataQualityEvaluator:
    def __init__(self, config, thresholds):
        self.config = config
        self.thresholds = thresholds
        self.quality_metrics = {}

    def evaluate(self, batch_id, df):
        metrics = {
            "batch_id": batch_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "quality_score": 0.0,
            "issues": []
        }

        null_ratios = df.isnull().sum() / len(df)
        metrics["null_ratios"] = null_ratios.to_dict()

        for col, ratio in null_ratios.items():
            if ratio > self.thresholds["max_null_ratio"]:
                metrics["issues"].append({
                    "type": "high_null_ratio",
                    "column": col,
                    "value": ratio,
                    "threshold": self.thresholds["max_null_ratio"]
                })

        numeric_cols = self.config.get("numeric_columns", [])
        outlier_ratios = {}

        for col in numeric_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                outlier_ratio = outliers / len(df)
                outlier_ratios[col] = outlier_ratio

                if outlier_ratio > self.thresholds["max_outlier_ratio"]:
                    metrics["issues"].append({
                        "type": "high_outlier_ratio",
                        "column": col,
                        "value": outlier_ratio,
                        "threshold": self.thresholds["max_outlier_ratio"]
                    })

        metrics["outlier_ratios"] = outlier_ratios

        cat_cols = self.config.get("categorical_columns", [])
        for col in cat_cols:
            if col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < self.thresholds["min_unique_ratio"]:
                    metrics["issues"].append({
                        "type": "low_cardinality",
                        "column": col,
                        "value": unique_ratio,
                        "threshold": self.thresholds["min_unique_ratio"]
                    })

        issue_count = len(metrics["issues"])
        metrics["quality_score"] = max(0, 100 - issue_count * 2)

        self.quality_metrics[batch_id] = metrics

        logger.info(f"Качество батча {batch_id}: {metrics['quality_score']} ({issue_count} проблем)")

        return metrics
    
class DataCleaner:
    def __init__(self, cleaning_config):
        self.config = cleaning_config

    def clean(self, df):
        df_clean = df.copy()
        cleaning_log = {"rows_removed": 0}

        before_rows = len(df_clean)
        df_clean.drop_duplicates(subset=["INSR_BEGIN", "OBJECT_ID"], inplace=True)
        cleaning_log["rows_removed"] = before_rows - len(df_clean)

        logger.info(f"Очистка завершена: удалено {cleaning_log['rows_removed']} дубликатов")

        return df_clean, cleaning_log

class AutoEDA:
    def __init__(self, reports_dir, max_categories=15):
        self.reports_dir = reports_dir
        self.max_categories = max_categories

    def generate(self, batch_id, df):
        eda_summary = {
            "batch_id": batch_id,
            "basic_stats": {},
            "correlations": {},
            "missing_patterns": {},
            "outliers_summary": {}
        }

        numeric_cols = df.select_dtypes(include=['number']).columns
        eda_summary["basic_stats"] = df[numeric_cols].describe().to_dict()

        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j]
                    })
            corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            eda_summary["correlations"] = corr_pairs[:5]

        eda_summary["missing_patterns"] = df.isnull().sum().to_dict()

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            eda_summary["outliers_summary"][col] = {
                "count": len(outliers),
                "ratio": len(outliers) / len(df),
                "lower_bound": lower,
                "upper_bound": upper
            }

        report_file = self.reports_dir / f"eda_batch_{batch_id:05d}.json"
        with open(report_file, "w") as f:
            json.dump(eda_summary, f, indent=2)

        self._create_plots(df, batch_id)

        logger.info(f"EDA выполнено для батча {batch_id}")
        return eda_summary

    def _create_plots(self, df, batch_id):
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            for idx, col in enumerate(numeric_cols[:4]):
                ax = axes[idx // 2, idx % 2]
                df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(f'Распраделение {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Частота')

            plt.tight_layout()
            plot_file = self.reports_dir / f"eda_batch_{batch_id:05d}_plots.png"
            plt.savefig(plot_file)
            plt.close()

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def create_features(self, df):
        df_feat = df.copy()
        features_log = []

        df_feat["INSR_BEGIN"] = pd.to_datetime(df_feat["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
        df_feat["INSR_END"] = pd.to_datetime(df_feat["INSR_END"], format="%d-%b-%y", errors="coerce")

        if "policy_duration_days" in self.config["features"]:
            df_feat["policy_duration_days"] = (df_feat["INSR_END"] - df_feat["INSR_BEGIN"]).dt.days
            df_feat["policy_duration_days"] = df_feat["policy_duration_days"].fillna(365)
            features_log.append("policy_duration_days")

        if "premium_per_seat" in self.config["features"]:
            df_feat["premium_per_seat"] = df_feat["PREMIUM"] / (df_feat["SEATS_NUM"] + 1e-6)
            features_log.append("premium_per_seat")

        if "insured_value_per_ton" in self.config["features"]:
            df_feat["insured_value_per_ton"] = df_feat["INSURED_VALUE"] / (df_feat["CCM_TON"] + 1e-6)
            features_log.append("insured_value_per_ton")

        if "claim_ratio" in self.config["features"]:
            df_feat["claim_ratio"] = df_feat["CLAIM_PAID"] / (df_feat["PREMIUM"] + 1e-6)
            df_feat["claim_ratio"] = df_feat["claim_ratio"].clip(0, 10)
            features_log.append("claim_ratio")

        if "is_claim" in self.config["features"]:
            df_feat["is_claim"] = (df_feat["CLAIM_PAID"] > 0).astype(int)
            features_log.append("is_claim")

        if "premium_log" in self.config["features"]:
            df_feat["premium_log"] = np.log(df_feat["PREMIUM"] + 1e-6)
            features_log.append("premium_log")

        logger.info(f"Создано {len(features_log)} новых признаков: {features_log}")
        return df_feat, features_log
    
class DataDriftDetector:
    def __init__(self, drift_config):
        self.config = drift_config
        self.reference_stats = None
        self.drift_history = []

    def set_reference(self, batches_df_list):
        if len(batches_df_list) == 0:
            logger.warning("Список контрольных батчей пуст")
            return

        combined = pd.concat(batches_df_list, ignore_index=True)
        self.reference_stats = {}

        for col in self.config["monitor_columns"]:
            if col in combined.columns:
                if combined[col].dtype in ['float64', 'int64']:
                    hist, bins = np.histogram(combined[col].dropna(), bins=10, density=True)
                    self.reference_stats[col] = {
                        "type": "numeric",
                        "histogram": hist.tolist(),
                        "bins": bins.tolist(),
                        "mean": float(combined[col].mean()),
                        "std": float(combined[col].std())
                    }
                else:
                    value_counts = combined[col].value_counts(normalize=True)
                    self.reference_stats[col] = {
                        "type": "categorical",
                        "distribution": value_counts.to_dict()
                    }

        logger.info(f"Reference stats set for {len(self.reference_stats)} columns")

    def detect_drift(self, df, batch_id):
        if self.reference_stats is None:
            logger.warning("Контрольная статистика пуста, пропуск этапа drift detection")
            return {}

        drift_report = {"batch_id": batch_id, "drifts": [], "psi_values": {}}

        for col in self.config["monitor_columns"]:
            if col not in df.columns or col not in self.reference_stats:
                continue

            ref = self.reference_stats[col]
            psi = self._calculate_psi(df[col], ref)
            drift_report["psi_values"][col] = psi

            if psi > self.config["psi_threshold"]:
                drift_report["drifts"].append({
                    "column": col,
                    "psi": psi,
                    "threshold": self.config["psi_threshold"],
                    "severity": "high" if psi > 0.3 else "moderate"
                })

        if drift_report["drifts"]:
            logger.warning(f"Обнаружен drift в батче {batch_id}: {drift_report['drifts']}")
        else:
            logger.info(f"В батче {batch_id} не обнаружен drift")

        self.drift_history.append(drift_report)
        return drift_report

    def _calculate_psi(self, current_series, reference):
        if reference["type"] == "numeric":
            hist_current, _ = np.histogram(
                current_series.dropna(), 
                bins=reference["bins"], 
                density=True
            )
            hist_current = np.array(hist_current) + 1e-6
            hist_ref = np.array(reference["histogram"]) + 1e-6

            psi = np.sum((hist_current - hist_ref) * np.log(hist_current / hist_ref))
            return float(psi)
        else:
            current_dist = current_series.value_counts(normalize=True).to_dict()
            psi = 0.0
            for category, ref_p in reference["distribution"].items():
                curr_p = current_dist.get(category, 1e-6)
                psi += (curr_p - ref_p) * np.log(curr_p / ref_p)
            return psi
        
def analyse_data():
    logger.info("Data Analysis начато")

    reports_dir = Path(CONFIG["logging"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_storage = DatabaseStorage(CONFIG["storage"]["raw_table"])
    cleaned_storage = DatabaseStorage(CONFIG["storage"]["cleaned_table"])
    quality_eval = DataQualityEvaluator(CONFIG["data_analysis"]["quality"], CONFIG["data_analysis"]["quality"]["thresholds"])
    cleaner = DataCleaner(CONFIG["data_analysis"]["cleaning"])
    eda = AutoEDA(reports_dir, CONFIG["data_analysis"]["eda"]["max_categories"])
    feature_eng = FeatureEngineer(CONFIG["data_analysis"]["feature_engineering"])
    drift_detector = DataDriftDetector(CONFIG["data_analysis"]["drift_detection"])

    quality_history = {}
    reference_batches = []

    ref_limit = CONFIG["data_analysis"]["drift_detection"]["reference_batches"]
    batch_id = 0

    for df_raw in raw_storage.read():
        quality_metrics = quality_eval.evaluate(batch_id, df_raw)
        quality_history[batch_id] = quality_metrics

        df_cleaned, cleaning_log = cleaner.clean(df_raw)

        if CONFIG["data_analysis"]["eda"]["enabled"]:
            eda.generate(batch_id, df_cleaned)

        if CONFIG["data_analysis"]["feature_engineering"]["enabled"]:
            df_features, features_list = feature_eng.create_features(df_cleaned)
        else:
            df_features = df_cleaned

        cleaned_storage.save_batch(batch_id, df_features, {"data_analysis_version": VERSION})

        if CONFIG["data_analysis"]["drift_detection"]["enabled"]:
            if batch_id < ref_limit:
                reference_batches.append(df_features)
                if batch_id + 1 == ref_limit:
                    drift_detector.set_reference(reference_batches)
            else:
                drift_report = drift_detector.detect_drift(df_features, batch_id)
                if drift_report["drifts"]:
                    logger.warning(f"Drift обнаружен в батче {batch_id}")

        batch_id += 1
        

    html_content = """<html>
<head><title>Data Quality Report - Ethiopian Insurance</title></head>
<body>
<h1>Data Quality Monitoring Report</h1>
<p>Generated: {timestamp}</p>
<table border="1">
<tr><th>Batch ID</th><th>Quality Score</th><th>Issues Count</th><th>Issues Details</th></tr>
{rows}
</table>
</body>
</html>"""
    
    rows_html = ""
    for batch_id, metrics in quality_history.items():
        issues_html = "<br>".join([f"{i['type']}: {i['column']} (value={i['value']:.3f})" for i in metrics.get("issues", [])])
        rows_html += f"""<tr>
    <td>{batch_id}</td>
    <td>{metrics['quality_score']}</td>
    <td>{len(metrics.get('issues', []))}</td>
    <td>{issues_html}</td>
</tr>"""
    
    report_file = reports_dir / f"quality_report_{datetime.now():%Y%m%d_%H%M%S}.html"
    with open(report_file, "w") as f:
        f.write(html_content.format(timestamp=datetime.now().isoformat(), rows=rows_html))
    
    logger.info(f"Очет о качестве данных сохранен в {report_file}")

    logger.info("Data Analysis закончено")

if __name__ == "__main__":
    main()