# Hướng dẫn chạy dự án

Bản tiếng Việt, rút gọn, của mục *Quickstart* và *Reproduce* trong
[`README.md`](../README.md). Giải thích chi tiết về MDP, giao thức đánh giá và
cách mở rộng nằm ở `README.md` và [`docs/extending.md`](../docs/extending.md).

**Mọi lệnh đều chạy từ thư mục gốc của dự án** (thư mục chứa `pyproject.toml`).
Đường dẫn trong `settings/*.yaml` là tương đối so với thư mục gốc.

## 1. Cài đặt (một lần)

Cần cài SUMO và đặt biến môi trường `SUMO_HOME`. Sau đó:

```bash
pip install -e ".[dev]"
pytest -q                                  # không cần SUMO
python comparison/fixed_phase_check.py     # cần SUMO; phải in PASS
```

Khi không bật GUI, SUMO chạy ngay trong tiến trình Python (`libsumo`), nhanh hơn
TraCI qua socket khoảng 3 lần, và cho kết quả giống hệt.

## 2. Thử nhanh (vài phút)

```bash
python src/train.py --episodes 10 --out runs/train/smoke
python src/evaluate.py dqn --model runs/train/smoke --episodes 5
python src/evaluate.py max_pressure --episodes 5
```

Hoặc mở `notebook/quickstart.ipynb`.

## 3. Quy trình đầy đủ

```bash
# Huấn luyện 3 lần với 3 seed khác nhau (có thể chạy song song)
python src/train.py --seed 0 --threads 2 --out runs/train/dqn-seed0
python src/train.py --seed 1 --threads 2 --out runs/train/dqn-seed1
python src/train.py --seed 2 --threads 2 --out runs/train/dqn-seed2

# Đánh giá greedy (ε = 0) trên 100 seed held-out 10000–10099
python src/evaluate.py dqn --model runs/train/dqn-seed0
python src/evaluate.py dqn --model runs/train/dqn-seed1
python src/evaluate.py dqn --model runs/train/dqn-seed2

# Baseline trên đúng các seed đó
python src/evaluate.py fixed_time --green 30
python src/evaluate.py fixed_time --green 10
python src/evaluate.py max_pressure
python src/evaluate.py random

# So sánh theo cặp (cùng seed = cùng dòng xe), có khoảng tin cậy 95%
python comparison/compare_results.py runs/eval/fixed_time-30s runs/eval/fixed_time-10s \
    runs/eval/max_pressure runs/eval/random \
    runs/eval/dqn-seed0 runs/eval/dqn-seed1 runs/eval/dqn-seed2 --out runs/comparison
```

## 4. File xuất ra

| Đường dẫn | Nội dung |
|---|---|
| `runs/train/<tên>/trained_model.pt` | Mô hình sau episode cuối, là mô hình được đánh giá |
| `runs/train/<tên>/training_settings.yaml` | Cấu hình thực sự đã dùng (kể cả `--seed`, `--episodes`) |
| `runs/train/<tên>/training_log.csv`, `training_curves.png` | Một dòng cho mỗi episode huấn luyện |
| `runs/eval/<nhãn>/eval_results.json` | Số liệu từng episode trên các seed held-out |
| `runs/comparison/summary.md`, `comparison.png` | Bảng và biểu đồ so sánh |

`runs/` không được commit. Một lần chạy được trích dẫn trong `README.md` được
chép vào `results/<tên>/` (xem [`results/README.md`](../results/README.md)).

## 5. Lưu ý

- **Chỉ so sánh bằng `total_delay`** (xe·giây, định nghĩa trong `src/metrics.py`).
  Tổng reward chỉ là chẩn đoán huấn luyện, không dùng để so với baseline.
- Huấn luyện có seed: cùng `--seed` và cùng `--threads` trên CPU cho kết quả
  giống hệt từng bit.
- Seed giao thông dùng để huấn luyện là `0 … total_episodes − 1`; seed đánh giá
  bắt đầu từ `10000`. Đừng chọn siêu tham số hay checkpoint dựa trên seed đánh giá.
- `gui: true` trong file settings sẽ mở `sumo-gui` (dùng TraCI, chậm hơn).
