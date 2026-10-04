# Explainable Machine Learning for Electricity Carbon Intensity

Repository tái lập cho luận văn thạc sĩ:

> **Ứng dụng học máy có khả năng giải thích trong mô hình hóa cường độ carbon của hệ thống điện các quốc gia dưới tác động của quá trình chuyển dịch cơ cấu nguồn điện**

## Kết quả đã chạy

Lần chạy đã xác minh với OWID Grapher API tạo panel **5.267 quốc gia–năm, 210 quốc gia, giai đoạn 2000–2025**. CatBoost được chọn trước khi xem test bằng RMSE validation thấp nhất. Trên temporal test 2022–2025, mô hình đạt MAE **24,68**, RMSE **41,04 gCO₂e/kWh**, R² **0,972**; trên unseen-country test đạt MAE **60,70**, RMSE **110,61**, R² **0,749**.

Đây là kết quả của pipeline chạy trên CSV mới tải trực tiếp từ Grapher, không phải số nhập tay. Nguồn chuẩn là `outputs/thesis/tables/*.csv`; paper chỉ nạp macro/bảng sinh từ các file đó.

## Dữ liệu và thiết kế

- Target: OWID Grapher `electricity-mix` (`source=total`, `metric=carbon_intensity`), nguồn Ember (2026), đơn vị gCO₂e/kWh vòng đời.
- Đầu vào tải trực tiếp từ OWID Grapher CSV API: cơ cấu phát điện, tổng phát điện, nhu cầu, nhập ròng, điện bình quân đầu người, dân số và GDP/người. Tỷ trọng từng nguồn được chuẩn hóa từ sản lượng Grapher chia tổng sản lượng; không cần tải/đặt thủ công bộ `owid-energy-data.csv`.
- Đặc trưng: cơ cấu coal/gas/oil/solar/wind/hydro/nuclear/renewables, generation, demand, imports, population và các biến dẫn xuất đã mô tả trong `docs/methodology_protocol.md`.
- Loại các aggregate OWID; giữ quốc gia có mã ISO và Kosovo.
- Temporal split: 2000–2018 / 2019–2021 / 2022–2025.
- Unseen-country: quốc gia độc quyền giữa train/validation/test.
- Region-transfer: giữ lại từng châu lục, test giai đoạn 2022–2025.
- Mô hình: Linear Regression, Ridge, Elastic Net, Random Forest, XGBoost, CatBoost.
- XAI: SHAP, permutation importance, partial dependence.
- Scenario là phản thực mô hình ceteris paribus, không phải ước lượng nhân quả.

## Cấu trúc repository

```text
data/
  raw/                         # CSV tải từ OWID Grapher API + metadata
  processed/                   # panel country-year đã kiểm tra
docs/
  methodology_protocol.md      # giao thức, leakage và giới hạn diễn giải
  literature_screening.csv     # tài liệu đã xác minh
notebooks/
  Explainable_Carbon_Intensity_Thesis.ipynb  # notebook thực nghiệm end-to-end
src/
  thesis_workflow.py           # pipeline chuẩn của luận văn
  carbon_intensity_workflow.py # khai báo URL Grapher, tải và chuẩn hóa CSV
scripts/
  download_owid_data.py        # tải mới nguồn CSV trực tiếp từ Grapher
  build_carbon_notebook.py     # sinh notebook phân tích cũ khi cần
outputs/thesis/
  figures/                     # 12 hình sinh tự động
  tables/                      # audit, metrics, XAI, Việt Nam, scenario
  models/                      # pipeline đã fit
  run_manifest.json            # phiên bản, URL, seed và SHA-256
outputs/kmeans/
  kmeans_owid_outputs.zip      # archive kết quả demo K-Means
paper/
  thesis_explainable_carbon_intensity.tex
  thesis_explainable_carbon_intensity.pdf
requirements.txt
run_all.ps1
run_all.sh
```

Code thực thi được giữ trong `src/`, script tiện ích/entrypoint trong `scripts/`,
nghiên cứu trong `notebooks/`, dữ liệu nguồn và panel trong `data/`, và kết quả
sinh tự động trong `outputs/`. Các notebook và artifact K-Means được giữ nguyên;
workflow luận văn chính dùng `thesis_workflow.py` và `outputs/thesis/`.

## Chạy trên Windows PowerShell

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1800 `
  .\notebooks\Explainable_Carbon_Intensity_Thesis.ipynb
```

Hoặc chạy toàn bộ bằng:

```powershell
.\run_all.ps1
```

Luồng notebook tải các CSV từ URL OWID Grapher API được khai báo trong
`src/carbon_intensity_workflow.py` khi chạy; không cần chuẩn bị dataset raw bằng
tay. Chọn kernel Jupyter là `.venv` vừa tạo và cài `requirements.txt` trước khi
chạy notebook.

## Google Colab

Upload/clone cả repository, chạy `pip install -r requirements.txt`, mở notebook
và chọn kernel đã cài các dependency đó, rồi **Run all**. Notebook tự tải CSV
trực tiếp từ OWID Grapher API khi chạy. Các file raw và SHA-256 được lưu lại để
ghi nhận chính xác dữ liệu đã dùng; lần chạy sau sẽ lấy dữ liệu mới nhất.

## Kiểm tra bằng chứng

- `data_audit.json`: số dòng, quốc gia, outlier và quy tắc lọc.
- `leakage_and_redundancy_audit.csv`: biến loại khỏi ma trận mô hình và lý do.
- `vif_train.csv`: đa cộng tuyến chỉ đo trên temporal train.
- `hyperparameter_search.csv`: toàn bộ validation search.
- `model_metrics.csv`: temporal và unseen-country test.
- `region_transfer_metrics.csv`: chuyển giao từng vùng.
- `bootstrap_confidence_intervals.csv`: CI 95% theo cụm quốc gia.
- `shap_importance.csv`, `permutation_importance.csv`, `partial_dependence.csv`: XAI.
- `vietnam_temporal_case.csv`, `vietnam_scenarios.csv`: nghiên cứu tình huống.
- `run_manifest.json`: provenance môi trường và SHA-256 dữ liệu.

## Nguồn chính

- [OWID — Carbon intensity of electricity](https://ourworldindata.org/grapher/carbon-intensity-electricity)
- [OWID Grapher — Electricity generation](https://ourworldindata.org/grapher/electricity-mix?source=total&metric=by_source&frequency=annual)
- [OWID Grapher — Electricity demand](https://ourworldindata.org/grapher/electricity-mix?source=total&metric=demand&frequency=annual)
- [OWID Grapher — Lifecycle carbon intensity](https://ourworldindata.org/grapher/electricity-mix?source=total&metric=carbon_intensity&frequency=annual)
- [OWID Grapher — Population](https://ourworldindata.org/grapher/population)
- [OWID Grapher — GDP per capita](https://ourworldindata.org/grapher/gdp-per-capita-worldbank)
