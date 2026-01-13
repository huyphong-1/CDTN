# CHUONG 1. ?NG D?NG MACHINE LEARNING

## 1.1. Gi?i thi?u bài toán
Sau khi hoàn thi?n h? th?ng ETL và Data Warehouse, d? li?u c?a doanh nghi?p dã du?c chu?n hóa, tích h?p và s?n sàng cho các ho?t d?ng phân tích nâng cao. Trong chuong này, Machine Learning du?c tri?n khai nhu m?t bu?c phát tri?n ti?p theo nh?m khai thác giá tr? d? doán t? d? li?u l?ch s?, thay vì ch? d?ng l?i ? phân tích mô t? và báo cáo.

Trong ph?m vi d? tài, nhóm t?p trung xây d?ng và tri?n khai hai mô hình Machine Learning:
- Mô hình d? doán t?n kho cu?i k? theo quý.
- Mô hình d? doán doanh thu theo quý.

Hai mô hình này ph?c v? tr?c ti?p cho các bài toán nghi?p v? c?t lõi: qu?n lý t?n kho và dánh giá hi?u qu? kinh doanh theo th?i gian.

## 1.2. Xác d?nh bài toán

### 1.2.1. B?i c?nh nghi?p v?
Doanh nghi?p c?n d? báo t?n kho d? t?i uu k? ho?ch nh?p hàng, h?n ch? thi?u h?t ho?c t?n du; d?ng th?i c?n d? báo doanh thu d? theo dõi xu hu?ng kinh doanh, h? tr? l?p k? ho?ch và dánh giá hi?u su?t theo quý. D? li?u du?c luu tr? trong Data Warehouse giúp d?m b?o tính nh?t quán và d? tin c?y cho vi?c hu?n luy?n mô hình.

### 1.2.2. Lo?i bài toán và m?c tiêu d? doán
- Lo?i bài toán: h?i quy chu?i th?i gian (time-series regression) ? m?c quý.
- M?c tiêu d? doán:
  - Doanh thu theo quý (QuarterRevenue).
  - T?n kho cu?i k? theo quý (QuarterEndInventory).

Bài toán du?c gi?i quy?t theo hu?ng d? báo m?t bu?c (next quarter) trong backtest và d? báo nhi?u bu?c trong tuong lai d?a trên s? quý c?u hình (forecast_quarters).

### 1.2.3. Ð?u vào và d?u ra c?a mô hình
**Ngu?n d? li?u chính** (SQL Server Data Warehouse):
- `FactSales`: dùng d? tính doanh thu theo quý v?i công th?c: `Revenue = LineAmount - DiscountAmount`.
- `FactInventorySnapshot`: dùng d? tính t?n kho cu?i k? theo quý b?ng cách l?y `ClosingQty` t?i ngày snapshot cu?i cùng c?a m?i quý (theo t?ng SKU) r?i c?ng t?ng.
- `DimDate`: ánh x? `DateKey` sang ngày và quý.

**Ð?u vào mô hình** (feature engineering ? m?c quý):
- Lags 1–4 quý c?a bi?n m?c tiêu.
- Trung bình tru?t 4 quý (Mean4).
- T?c d? tang tru?ng theo quý (Growth1).
- Y?u t? mùa v? theo quý: `q_sin`, `q_cos`.
- Nhãn th?i gian: `year`, `q`.

**Ð?u ra mô hình**:
- D? báo doanh thu theo quý (CSV + bi?u d?).
- D? báo t?n kho cu?i k? theo quý (CSV + bi?u d?).

## 1.3. L?a ch?n mô hình và hu?n luy?n
Mô hình du?c ch?n là **Linear Regression** do:
- D? li?u theo quý có s? lu?ng di?m không l?n, mô hình tuy?n tính giúp tránh overfitting.
- D? gi?i thích và nhanh tri?n khai trong môi tru?ng th?c t?.
- Phù h?p d? làm baseline v?ng ch?c tru?c khi m? r?ng sang mô hình ph?c t?p hon.

Quy trình hu?n luy?n:
1. T?ng h?p d? li?u theo quý t? DWH.
2. T?o d?c trung (lags, rolling mean, growth, seasonality).
3. Làm s?ch d? li?u (lo?i b? NaN do t?o lag).
4. Hu?n luy?n mô hình trên toàn b? chu?i l?ch s?.
5. D? báo nhi?u bu?c trong tuong lai b?ng cách s? d?ng d? báo c?a bu?c tru?c làm d?u vào cho bu?c sau (multi-step forecasting).

## 1.4. Ðánh giá mô hình

### 1.4.1. Tiêu chí dánh giá
Các ch? s? du?c s? d?ng:
- **MAE** (Mean Absolute Error): sai s? tuy?t d?i trung bình.
- **RMSE** (Root Mean Squared Error): nh?n m?nh sai s? l?n.
- **MAPE** (Mean Absolute Percentage Error): sai s? tuong d?i.
- **R2**: m?c d? gi?i thích phuong sai.

Ð? so sánh, h? th?ng có baseline don gi?n: d? báo quý ti?p theo b?ng giá tr? c?a quý tru?c dó (Lag1).

### 1.4.2. K?t qu? dánh giá và so sánh mô hình
K?t qu? dánh giá du?c luu t?i:
- Doanh thu: `outputs/revenue_backtest_quarterly.csv`, `outputs/revenue_linear_metrics.csv`
- T?n kho: `outputs/inventory_backtest_quarterly.csv`, `outputs/inventory_linear_metrics.csv`

Các bi?u d? minh h?a:
- `outputs/revenue_actual_vs_pred_backtest.png`
- `outputs/inventory_actual_vs_pred_backtest.png`

Nhìn chung, mô hình tuy?n tính cho k?t qu? ?n d?nh và có th? gi?i thích du?c trong b?i c?nh d? li?u theo quý. K?t qu? d? báo du?c hi?n th? trên dashboard n?i b? d? h? tr? phân tích và ra quy?t d?nh.
