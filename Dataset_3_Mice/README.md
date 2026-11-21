# Dataset 3 — Thời gian sống của chuột

- Dữ liệu: [data/mice_survival.csv](data/mice_survival.csv).
- Nhóm `Treatment`: **7 chuột**; nhóm `Control`: **9 chuột**; tổng cộng **16 chuột**.
- `Survival_Days`: thời gian sống, đơn vị ngày được ghi ở slide.
- `Mouse_ID` chỉ là mã hàng được thêm để nhận diện quan sát, không phải mã cá thể gốc.
- Nguồn: ví dụ (iii), slide 5/71, AS3179 Lecture 08; số liệu trùng mảng `trt`, `ctl`
  trong `AS3179_Lecture08_thuc_hanh.py`. Slide dẫn Efron và Tibshirani (1993).
- **Chạy:** `python data3.py` với Python đã cài NumPy, SciPy, Pandas và Matplotlib.
- **Trạng thái:** có phân tích cơ bản bằng `data3.py`; thống kê mô tả, chênh lệch trung bình,
  Welch và CI xấp xỉ, hoán vị chính xác, hoán vị Monte Carlo và MCSE. Đầu ra lưu trong `results/`,
  gồm CSV, JSON, hình 300 DPI và `interpretation.md`. Chưa kiểm chứng đầy đủ Mục 1, 2, 5 cho Dataset 3.

Bộ này có hai nhóm, khác Dataset 1. Muốn kiểm định hoán vị cần nêu rõ giả thuyết không
và cơ sở trao đổi nhãn/ngẫu nhiên hóa; chỉ có nhãn nhóm chưa tự chứng minh điều kiện đó.
Không tự bổ sung giả định về thiết kế thử nghiệm hoặc kiểm duyệt khi chưa có thông tin.
