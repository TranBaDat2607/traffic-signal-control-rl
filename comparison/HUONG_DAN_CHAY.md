# Hướng Dẫn Chạy Dự Án Traffic Signal Reinforcement Learning

> **Mô tả:** Agent DQN điều khiển đèn tín hiệu của một ngã tư trong SUMO. Dưới
> đây là đúng quy trình đã tạo ra kết quả công bố trong `README.md`: baseline
> fixed-time, huấn luyện, đánh giá greedy trên các seed held-out, rồi vẽ biểu đồ
> so sánh. Bản tiếng Anh nằm ở mục *Reproduce* của `README.md`.

**Mọi lệnh đều chạy từ thư mục gốc của dự án** (thư mục chứa `pyproject.toml`).
Đường dẫn trong `settings/*.yaml` và trong file cấu hình SUMO là tương đối so
với thư mục gốc.

Chỉ chạy **một** lệnh có SUMO tại một thời điểm. Mỗi episode đều ghi đè
`intersection/episode_routes.rou.xml`, và SUMO đọc file đó dần trong lúc chạy.

---

## 1. Cài đặt (một lần)

Cần cài SUMO và đặt biến môi trường `SUMO_HOME`. Sau đó:

```bash
pip install -e ".[dev]"
```

Lệnh này cài cả `libsumo`: khi không bật GUI, SUMO chạy ngay trong tiến trình
Python, nhanh hơn TraCI qua socket khoảng 3 lần. Kết quả giống hệt nhau
(PROBLEMS.md P3-1).

---

## 2. Quy trình chạy (Pipeline)

### Bước 1: Baseline fixed-time trên các seed held-out

SUMO tự chạy kế hoạch pha trong `intersection/environment.net.xml` (30 s xanh /
4 s vàng). Python không điều khiển đèn, chỉ đo. Dùng `settings/testing_settings.yaml`
để baseline chạy trên **cùng các seed** mà agent được đánh giá (`episode_seed:
10000`, 100 episode). Các seed này không trùng với seed huấn luyện (0…299).

```bash
python comparison/run_baseline.py --config settings/testing_settings.yaml --out baseline_results
```

- **File xuất ra:** `baseline_results/baseline_results.json` (kèm danh sách seed và
  thời lượng các pha thực sự đã chạy) và các file `.txt`.

### Bước 2: Huấn luyện agent DQN

```bash
python src/train.py --settings settings/training_settings.yaml --out model/dqn-300ep
```

- 300 episode, ε giảm tuyến tính từ 1 về 0. Trên máy 12 nhân, chỉ CPU, lần chạy
  công bố mất 35 phút.
- **File xuất ra:** `model/dqn-300ep/trained_model.pt` (mô hình cuối, là mô hình
  được đánh giá ở bước 3), `best_model.pt`, các file `plot_*_data.txt` và ảnh
  `plot_*.png` của đường cong huấn luyện.
- `plot_reward_*` chỉ là chẩn đoán huấn luyện, **không** dùng để so với baseline
  (PROBLEMS.md P0-1).

### Bước 3: Đánh giá greedy (ε = 0) trên các seed held-out

```bash
python src/test.py --model model/dqn-300ep
```

- **File xuất ra:** `model/dqn-300ep/test/test_results.json` và các file
  `plot_*_data.txt`.

### Bước 4: So sánh với baseline

```bash
python comparison/compare_results.py --baseline baseline_results --rl model/dqn-300ep/test --out comparison_output
```

- Script kiểm tra rằng hai bên chạy trên cùng các seed. Nếu không, nó dừng và
  báo lỗi.
- **File xuất ra:** `comparison_output/comparison.png` và
  `comparison_output/comparison_summary.json` (trung bình, chênh lệch ghép cặp
  theo seed, khoảng tin cậy 95%, số episode agent tốt hơn). Bảng tóm tắt cũng
  được in ra console.
- Thấp hơn là tốt hơn cho delay và queue. `Insertion delay` là số xe-giây phải
  chờ để vào được mạng lưới. Phần này **không** nằm trong total delay; nếu nó
  không nhỏ thì total delay chỉ là cận dưới (PROBLEMS.md P2-4).

### Bước 5 (tùy chọn): Baseline fixed-time với thời lượng pha của chính agent

Agent dùng 10 s xanh / 4 s vàng. Baseline này chạy một chu kỳ cố định với đúng
các thời lượng đó, để biết agent hơn hay kém một chu kỳ cố định cùng nhịp:

```bash
python comparison/run_baseline.py --config settings/testing_settings.yaml --out baseline_results_10-4 --green 10 --yellow 4
python comparison/compare_results.py --baseline baseline_results_10-4 --rl model/dqn-300ep/test --out comparison_output_10-4
```

### Kiểm tra phép đo

```bash
python comparison/fixed_phase_check.py
pytest -q
```

`fixed_phase_check.py` chạy cùng một chu kỳ đèn 10/4 qua hai đường: agent
(`Environment.execute()`) và SUMO tự chạy. Cả hai phải cho đúng 18,195
xe-giây (seed 0), in ra `PASS`.

---

## 3. Giao diện trực quan SUMO (GUI)

Mặc định mọi lệnh chạy ngầm (headless). Để xem xe chạy, thêm `--gui` vào lệnh
baseline (ví dụ một episode):

```bash
python comparison/run_baseline.py --config settings/testing_settings.yaml --out baseline_results_gui --episodes 1 --gui
```

Với huấn luyện/đánh giá: đặt `gui: true` trong file settings tương ứng. Chế độ
GUI dùng TraCI (không dùng được libsumo), nên chậm hơn.

> **Mẹo:** đặt tên thư mục khác nhau cho mỗi lần chạy (`model/run-02`,
> `model/run-03`, …). Mọi thư mục output mặc định đều bị `.gitignore` bỏ qua.
> Kết quả muốn công bố thì chép vào `results/<tên-run>/` (xem `results/README.md`).
