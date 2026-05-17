import datetime
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger

LOGGER = None
REPORTS_DIR = CONFIG["logging"]["reports_dir"]
HISTORY_FILE = os.path.join(REPORTS_DIR, "training_history.json")
DASHBOARD_FILE = os.path.join(REPORTS_DIR, "dashboard.html")


def build_html(history, best_model_data, validation_results, feature_importances):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_name = best_model_data.get("best_model", "—") if best_model_data else "—"
    best_mae = best_model_data.get("mae", "—") if best_model_data else "—"
    run_num = os.environ.get("GITHUB_RUN_NUMBER", "local")
    sha = os.environ.get("GITHUB_SHA", "")[:7] or "—"

    models_rows = ""
    if validation_results:
        for model_name, metrics in sorted(validation_results.items(), key=lambda x: x[1].get("MAE", float("inf"))):
            bold = "font-weight:bold" if model_name == best_name else ""
            models_rows += (
                f"<tr style='{bold}'><td>{model_name}</td>"
                f"<td>{metrics.get('MAE', '—')}</td>"
                f"<td>{metrics.get('RMSE', '—')}</td>"
                f"<td>{metrics.get('R2', '—')}</td></tr>\n"
            )

    feat_rows = ""
    if feature_importances:
        for feat, imp in list(feature_importances.items())[:15]:
            bar_w = int(float(imp) * 250)
            feat_rows += (
                f"<tr><td>{feat}</td><td>{float(imp):.4f}</td>"
                f"<td><div style='background:#4a90d9;height:10px;width:{bar_w}px'></div></td></tr>\n"
            )

    history_rows = ""
    for run in reversed(history):
        history_rows += (
            f"<tr><td>#{run.get('github_run_number', '—')}</td>"
            f"<td>{run.get('timestamp', '—')[:19]}</td>"
            f"<td>{run.get('github_sha', '—')}</td>"
            f"<td>{run.get('best_model', '—')}</td>"
            f"<td>{run.get('mae', '—')}</td>"
            f"<td>{run.get('rmse', '—')}</td>"
            f"<td>{run.get('r2', '—')}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Отчёт по обучению</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th {{ background: #555; color: #fff; padding: 6px 10px; text-align: left; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Отчёт по обучению</h1>
<p>Дата: {now} | Run: #{run_num} | SHA: {sha}</p>
<p>Лучшая модель: <b>{best_name}</b> (MAE={best_mae})</p>

<h2>Все модели</h2>
<table>
  <tr><th>Модель</th><th>MAE</th><th>RMSE</th><th>R2</th></tr>
  {models_rows if models_rows else "<tr><td colspan='4'>нет данных</td></tr>"}
</table>

<h2>Важность признаков (Decision Tree)</h2>
<table>
  <tr><th>Признак</th><th>Важность</th><th></th></tr>
  {feat_rows if feat_rows else "<tr><td colspan='3'>нет данных</td></tr>"}
</table>

<h2>История запусков</h2>
<table>
  <tr><th>Run</th><th>Дата</th><th>SHA</th><th>Лучшая модель</th><th>MAE</th><th>RMSE</th><th>R2</th></tr>
  {history_rows if history_rows else "<tr><td colspan='7'>нет данных</td></tr>"}
</table>
</body>
</html>"""
    return html


def write_github_summary(best_model_data, validation_results, history):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    best_name = best_model_data.get("best_model", "—") if best_model_data else "—"
    best_mae = best_model_data.get("mae", "—") if best_model_data else "—"

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"## Отчёт по обучению\n\n")
        f.write(f"Лучшая модель: **{best_name}**, MAE = **{best_mae}**\n\n")
        if validation_results:
            f.write("Все модели:\n\n")
            for name, m in sorted(validation_results.items(), key=lambda x: x[1].get("MAE", float("inf"))):
                f.write(f"- {name}: MAE={m.get('MAE', '—')}, RMSE={m.get('RMSE', '—')}, R2={m.get('R2', '—')}\n")
        if history:
            f.write(f"\nВсего запусков в истории: {len(history)}\n")


def generate_dashboard():
    global LOGGER
    LOGGER = setup_logger(
        "Dashboard",
        log_file=CONFIG["model_serving"]["log_file"],
        level=CONFIG["logging"]["level"],
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    best_model_data = None
    validation_results = None
    feature_importances = None

    best_model_path = os.path.join(REPORTS_DIR, "best_model.json")
    if os.path.exists(best_model_path):
        with open(best_model_path, "r") as f:
            best_model_data = json.load(f)

    validation_path = os.path.join(REPORTS_DIR, "validation_results.json")
    if os.path.exists(validation_path):
        with open(validation_path, "r") as f:
            validation_results = json.load(f)

    importances_path = os.path.join(REPORTS_DIR, "feature_importances.json")
    if os.path.exists(importances_path):
        with open(importances_path, "r") as f:
            feature_importances = json.load(f)

    # обновляем историю запусков
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

    if best_model_data is not None:
        best_name = best_model_data.get("best_model", "")
        best_metrics = validation_results.get(best_name, {}) if validation_results else {}
        entry = {
            "run_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.datetime.now().isoformat(),
            "github_run_number": os.environ.get("GITHUB_RUN_NUMBER", "local"),
            "github_sha": os.environ.get("GITHUB_SHA", "")[:7],
            "best_model": best_name,
            "mae": best_model_data.get("mae", best_metrics.get("MAE", 0)),
            "rmse": best_metrics.get("RMSE", 0),
            "r2": best_metrics.get("R2", 0),
            "all_models": validation_results or {},
        }
        history.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    html = build_html(history, best_model_data, validation_results, feature_importances)
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    LOGGER.info(f"dashboard сохранён: {DASHBOARD_FILE}")

    write_github_summary(best_model_data, validation_results, history)


if __name__ == "__main__":
    generate_dashboard()
