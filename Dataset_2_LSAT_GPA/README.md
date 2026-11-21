# Dataset 2 — Trường luật: LSAT và GPA

- Dữ liệu: [data/law_schools.csv](data/law_schools.csv).
- Có **15 trường**, mỗi hàng là **một cặp LSAT–GPA** của cùng trường.
- `School_ID` chỉ là mã hàng được thêm để nhận diện cặp, không phải biến đo gốc.
- Nguồn: ví dụ (ii), slide 5/71, AS3179 Lecture 08; số liệu trùng các mảng `lsat`, `gpa`
  trong `AS3179_Lecture08_thuc_hanh.py`. Slide dẫn Efron và Tibshirani (1993).
- Đại lượng nghiên cứu trong ví dụ gốc là tương quan giữa hai biến, không phải khoảng thời gian hỏng điều hòa.
- **Chạy:** `python data2.py` với Python đã cài NumPy, SciPy, Pandas và Matplotlib.
- **Trạng thái:** có phân tích cơ bản bằng `data2.py`; thống kê mô tả, tương quan Pearson,
  CI Fisher xấp xỉ và hoán vị kiểm định độc lập. Đầu ra lưu trong `results/`, gồm
  CSV, JSON, hình 300 DPI và `interpretation.md`. Chưa kiểm chứng đầy đủ Mục 1, 2, 5 cho Dataset 2.

Khi lấy mẫu các trường, phải giữ hai giá trị LSAT và GPA cùng hàng; không lấy mẫu hai cột độc lập.
Không áp dụng mặc định mô hình Exponential/Weibull của Dataset 1 lên bộ này.
Hoán vị GPA dùng H0 độc lập và giả định các trường iid/trao đổi; không chỉ H0 tương quan bằng 0.
CI Fisher dựa trên giả định chuẩn hai chiều; các giả định này chưa được hồ sơ lấy mẫu xác nhận.
