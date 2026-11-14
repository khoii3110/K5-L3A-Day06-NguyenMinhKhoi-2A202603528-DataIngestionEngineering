# Pipeline · PostgreSQL + pgvector

Pipeline cá nhân dùng PostgreSQL + pgvector local để chạy corpus PostgreSQL 17 cùng benchmark đã chuẩn bị. Cấu hình dùng embedding thật qua OpenAI API; không có vector giả hoặc fallback im lặng. Có thể thay vector store/model nếu ghi rõ và giữ cùng cấu hình giữa các chiến lược.

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # psycopg, pypdf và python-dotenv
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra kết nối và extension (đã chạy được)
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # TODO: đọc / chuẩn hóa tài liệu
│   ├── chunking.py        # TODO: chia chunk
│   ├── embeddings.py      # TODO: chọn model API hoặc local
│   ├── store.py           # TODO: lưu vector, metadata và filter
│   ├── evaluation.py      # TODO: chấm truy xuất, lưu trace
│   └── run.py             # TODO: kết nối các bước thành pipeline
├── data/                  # corpus được chuẩn bị ở ../data/
└── results/               # trace, metric và biểu mẫu audit được tạo khi chạy
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Cấu hình embedding

Trong `starter/.env`, đặt `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY`. Nếu chỉ đặt một key, pipeline tự nhận diện provider; nếu đặt cả hai, chỉ định `EMBEDDING_PROVIDER`. OpenRouter dùng `openai/text-embedding-3-small` tại `https://openrouter.ai/api/v1`, còn OpenAI trực tiếp dùng `text-embedding-3-small`. Cả hai mặc định 1536 chiều. Có thể đặt `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` và URL tương ứng nếu dùng model tương thích. Cấu hình trong `.env.example` minh họa OpenRouter. Chỉ gửi dữ liệu tới provider sau khi đã kiểm tra quyền sử dụng và chi phí; không commit `.env`. Có thể đặt `EMBEDDING_USD_PER_1M_TOKENS` theo bảng giá hiện tại để pipeline tính chi phí ước tính. Nếu không đặt, báo cáo sẽ ghi chi phí chưa được ước tính.

## 3. Corpus, benchmark và cấu hình

Corpus và manifest ở `../data/postgresql17_units.jsonl` và `../data/postgresql17_manifest.json`. Bộ câu hỏi 15 dev / 5 test ở `../benchmark/postgresql17_questions.jsonl`; các trích dẫn được kiểm tra tự động trước khi chạy.

Ba chiến lược được cố định trong pipeline:

| Strategy | Chunking | Filter |
|---|---|---|
| A_fixed | Cửa sổ 1.000 ký tự, overlap 100 | Không |
| B_paragraph | Gom theo đoạn, tối đa 1.000 ký tự | Không |
| C_paragraph_filter | Giống B | Lọc `chapter` khi câu hỏi có filter |

Mọi chiến lược dùng cùng corpus, model, benchmark và cosine distance. Tìm kiếm pgvector là exact (không tạo approximate index). C tái sử dụng vector của B vì chunk và embedding giống nhau; các hàng được lưu tách strategy. Nạp lại thay toàn bộ chunk cũ của strategy trong một transaction.

## 4. Chạy thử không gọi API

Từ `starter/`, xác minh corpus, gold và số lượng chunk mà không kết nối database
hay gọi dịch vụ embedding:

```powershell
python -m src.run --dry-run
```

## 5. Chạy benchmark thật

Sau khi đã cấu hình `OPENAI_API_KEY` và đồng ý với chi phí API, chạy từ `starter/`:

```powershell
python -m pip install -r requirements.txt
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
python -m src.run
```

Kết quả nằm trong `results/`: trace top-3 đầy đủ, tổng hợp Hit@1 / Evidence Hit@3 / filter compliance, metadata chạy, và ba bảng audit grounding, metadata, chunk coherence. Bảng audit dưới đây đã được điền bằng đánh giá hỗ trợ AI, dựa trên trace và đối chiếu corpus; đây không phải đánh giá độc lập của học viên. Học viên cần rà soát nhãn/lý do, sửa nếu không đồng ý và ghi rõ phương pháp/người đánh giá trong báo cáo. Không xem các nhãn này là ground truth hoặc nhãn của người chấm.

Audit đã điền:

| Strategy | Grounding score counts (0/1/2) | Grounding mean /2 | Coherence mean /2 (10 fixed positions) | Metadata correct | Filtered test questions compliant |
|---|---:|---:|---:|---:|---:|
| A_fixed | 1 / 2 / 2 | 1.2 | 0.9 | 15/15 | 3/3 |
| B_paragraph | 1 / 2 / 2 | 1.2 | 1.6 | 15/15 | 3/3 |
| C_paragraph_filter | 1 / 2 / 2 | 1.2 | 1.6 | 15/15 | 3/3 |

Grounding covers the same five test questions per strategy. Metadata checks cover all five test questions × top-3. Filter compliance is 3/3 filtered questions (9/9 returned chunks) per strategy.

Lượt chạy đầu tiên (2026-11-14, OpenRouter `openai/text-embedding-3-small`, 1536 chiều):

| Strategy | Dev Hit@1 | Dev Evidence Hit@3 | Test Hit@1 | Test Evidence Hit@3 | Chunks |
|---|---:|---:|---:|---:|---:|
| A_fixed | 5/15 | 12/15 | 2/5 | 2/5 | 179 |
| B_paragraph | 8/15 | 12/15 | 2/5 | 2/5 | 186 |
| C_paragraph_filter | 8/15 | 12/15 | 2/5 | 2/5 | 186 |

Test top-3 metadata khớp filter ở 3/3 câu có filter trong cả ba chiến lược; trên bộ mẫu này C không đổi top-3 so với B. Đây chỉ là kết quả retrieval tự động trên 5 câu test, không phải kết luận tổng quát hay nhãn grounding/coherence thủ công. Các CSV audit vẫn cần được người làm bài đọc, chấm và ghi lý do. Lượt chạy dùng 64.851 embedding tokens; chi phí chưa được ước tính.

Sau khi lưu kết quả, có thể dừng PostgreSQL nhưng vẫn giữ dữ liệu:

```powershell
docker compose -p day6_student_starter stop
```
