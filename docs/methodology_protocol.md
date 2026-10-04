# Giao thức phương pháp và chống rò rỉ dữ liệu

Ngày khóa giao thức: 2026-09-26. Seed cố định: 42.

## Câu hỏi nghiên cứu

1. Cơ cấu nguồn điện và quy mô hệ thống giải thích cường độ carbon điện ở mức nào?
2. Mô hình có tổng quát hóa sang tương lai, quốc gia chưa thấy và vùng chưa thấy hay không?
3. Đặc trưng nào chi phối dự báo toàn cục và trường hợp Việt Nam?
4. Dự báo mô hình thay đổi ra sao dưới các thay đổi cơ cấu điện giữ các biến khác không đổi?

## Đơn vị phân tích và target

- Đơn vị: quốc gia–năm, 2000–2025.
- Target: `carbon_intensity_gco2e_kwh`, cường độ phát thải vòng đời của điện theo OWID/Ember.
- Dữ liệu được tải trực tiếp khi chạy từ CSV endpoint chính thức của OWID Grapher: `electricity-mix` với các metric `carbon_intensity`, `by_source`, `generation`, `demand`, `net_imports`, `per_capita`; thêm Grapher `population` và `gdp-per-capita-worldbank`. Không dùng bản Energy Data tải riêng hoặc file dataset nhập thủ công.
- Tỷ trọng nhiên liệu được tính bằng sản lượng của nguồn trong metric `by_source` chia cho tổng sản lượng metric `generation`, nhân 100. Tỷ trọng nhập ròng bằng metric `net_imports` chia `demand`, nhân 100. GDP/người được giữ trong panel mô tả, không dùng làm biến mô hình.
- Chỉ giữ mã quốc gia thực; loại các aggregate `OWID_*`, ngoại trừ Kosovo (`OWID_KOS`).
- Hàng phải có ít nhất 9/11 trường đầu vào thô bắt buộc. Median imputation nằm bên trong pipeline và chỉ học từ train.

## Đặc tả đầu vào

Panel lưu toàn bộ coal, gas, oil, solar, wind, hydro, nuclear, renewables, biofuel, other renewables, generation, demand, imports, population, GDP và electricity per capita.

Ma trận mô hình dùng coal, gas, solar, wind, hydro, nuclear, biofuel; `oil_share_elec` là thành phần tham chiếu của dữ liệu thành phần. Lựa chọn thành phần tham chiếu được kiểm tra trên train: bỏ oil cho VIF tối đa thấp nhất. Demand và population dùng `log1p`; thêm net imports share và xu thế năm. Aggregate renewables, generation, per-capita electricity và GDP chỉ dùng mô tả vì dư thừa đại số, biến dẫn xuất hoặc thiếu dữ liệu lớn.

## Giao thức đánh giá

- Temporal: train 2000–2018, validation 2019–2021, test 2022–2025.
- Unseen-country: chia quốc gia độc quyền 60/20/20 bằng seed 42; không quốc gia nào xuất hiện ở hai split.
- Region-transfer: giữ lại lần lượt Africa, America, Asia, Europe, Oceania; train các vùng còn lại đến 2021 và test vùng giữ lại giai đoạn 2022–2025.

Siêu tham số được chọn bằng validation RMSE. Test không tham gia lựa chọn mô hình. Báo cáo MAE, RMSE và R². Khoảng tin cậy 95% dùng bootstrap theo cụm quốc gia 500 lần.

## Explainability và giới hạn diễn giải

SHAP, permutation importance và partial dependence được tính cho mô hình được chọn bằng temporal validation RMSE. Đây là giải thích của hàm dự báo, không chứng minh quan hệ nhân quả. Scenario Việt Nam là phản thực mô hình ceteris paribus, giữ demand, population, imports và year cố định; không phải dự báo chính sách.
