# Kế hoạch phân công: Tóm tắt báo cáo tài chính tiếng Việt

## 1. Mục tiêu và quyết định kiến trúc

### 1.1 Bối cảnh hiện tại

Repository hiện chưa có mã nguồn, dữ liệu hay quy trình kiểm thử. Nhóm cần xây dựng một hệ thống nghiên cứu có thể:

1. Đọc báo cáo tài chính quý/năm từ PDF hoặc dữ liệu text + bảng đã bóc tách, ưu tiên doanh nghiệp phi tài chính thuộc VN30/VN100 và có truy vết trang.
2. Huấn luyện từ đầu mô hình `BiLSTM + Attention` để chọn các câu hoặc đoạn quan trọng.
3. Fine-tune `BARTpho` làm mô hình sinh chính để tạo Executive Summary 250-400 từ theo cấu trúc cố định.
4. Đối chiếu mọi số liệu trong bản tóm tắt với báo cáo gốc.
5. Đánh giá riêng từng giai đoạn và đánh giá toàn bộ pipeline.

Lý do dùng hai giai đoạn là báo cáo tài chính dài hơn giới hạn đầu vào thông thường của mô hình sinh. Giai đoạn trích xuất giảm độ dài và giữ bằng chứng; giai đoạn sinh làm bản tóm tắt dễ đọc. Bộ kiểm chứng số liệu là cổng an toàn, không phải một chỉ số phụ.

### 1.2 Quyết định

Nhóm triển khai một **modular monolith bằng Python**, không tách microservice. Các module trao đổi qua schema có phiên bản và artifact trên hệ thống lưu trữ dùng chung. Cách này đủ cho đồ án 5 người, chạy được bằng CLI/notebook và tránh chi phí vận hành nhiều dịch vụ.

```text
PDF/HTML/TXT
    |
    v
Tiền xử lý + OCR + chuẩn hóa bảng  [TV1]
    |
    v
DocumentRecord v1
    |
    v
BiLSTM-Attention Extractor          [TV2]
    |
    v
EvidenceBundle v1
    |
    v
BARTpho Generator                   [TV3]
    |
    v
SummaryDraft v1
    |
    v
Numerical Verifier + Evaluation     [TV4]
    |
    v
VerifiedSummary v1 / REJECTED

CI, contract, review, merge, release [TV5]
```

Không huấn luyện đồng thời end-to-end trong phiên bản đầu. Mỗi giai đoạn phải có baseline và metric độc lập trước khi tích hợp.

## 2. Yêu cầu và phạm vi

### 2.1 Yêu cầu chức năng

| ID | Yêu cầu | Tiêu chí chấp nhận ban đầu |
|---|---|---|
| FR-01 | Nhập báo cáo tài chính tiếng Việt | Đọc PDF quý/năm hoặc text + bảng đã bóc tách của doanh nghiệp phi tài chính; corpus chính là VN30/VN100, PDF scan được OCR hoặc báo lỗi rõ ràng |
| FR-02 | Giữ khả năng truy vết | Mỗi câu/ô bảng có `document_id`, `page`, `source_span` hoặc bounding box nếu có |
| FR-03 | Trích xuất thông tin quan trọng | Xuất câu, điểm attention/salience và thứ tự gốc |
| FR-04 | Sinh tóm tắt | Fine-tune BARTpho và sinh Executive Summary tiếng Việt dài 250-400 từ |
| FR-05 | Kiểm chứng số liệu | Mỗi số trong tóm tắt được gắn bằng chứng hoặc bị đánh dấu `unsupported`/`conflict` |
| FR-06 | Đánh giá tái lập | Cấu hình, seed, phiên bản dữ liệu, commit và metric được lưu cùng mỗi lần chạy |
| FR-07 | Chạy end-to-end | Một lệnh nhận tài liệu và tạo kết quả JSON cùng bản tóm tắt dễ đọc |
| FR-08 | Tuân thủ cấu trúc đầu ra | Đủ và đúng thứ tự 4 mục: Kết quả kinh doanh; Tình hình tài chính & Dòng tiền; Nguyên nhân biến động chính; Điểm lưu ý / Rủi ro |

Nội dung đầu vào bắt buộc tập trung vào:

1. Bảng cân đối kế toán.
2. Báo cáo kết quả hoạt động kinh doanh.
3. Báo cáo lưu chuyển tiền tệ.
4. Công văn giải trình biến động lợi nhuận và các thuyết minh trọng yếu liên quan.

Nội dung đầu ra của bốn mục:

| Mục | Thông tin tối thiểu |
|---|---|
| Kết quả kinh doanh | Doanh thu thuần, lợi nhuận gộp/sau thuế, biên lợi nhuận và tăng/giảm YoY |
| Tình hình tài chính & Dòng tiền | Tổng tài sản, nợ phải trả/nợ vay và lưu chuyển tiền thuần từ hoạt động kinh doanh |
| Nguyên nhân biến động chính | Nguyên nhân được truy xuất từ giải trình hoặc thuyết minh, không tự suy diễn |
| Điểm lưu ý / Rủi ro | Các dấu hiệu như tồn kho/phải thu tăng mạnh, chi phí lãi vay tăng, dòng tiền kinh doanh âm hoặc bất thường có bằng chứng |

### 2.2 Ngoài phạm vi phiên bản đầu

- Không tư vấn đầu tư hoặc kết luận doanh nghiệp tốt/xấu.
- Không triển khai production API chịu tải Internet.
- Không bảo đảm xử lý mọi bố cục PDF scan chất lượng thấp.
- Không tự động sửa số liệu không khớp bằng suy đoán.
- Không huấn luyện một mô hình ngôn ngữ lớn từ đầu; chỉ BiLSTM-Attention được train from scratch.
- Không đưa ngân hàng, bảo hiểm và tổ chức tài chính vào corpus chính vì mẫu biểu khác doanh nghiệp phi tài chính.
- Không duy trì ViT5 như pipeline chính; chỉ dùng làm baseline nếu thời gian và tài nguyên còn đủ.

### 2.3 Ràng buộc chất lượng

| Nhóm | Mục tiêu/giả định ban đầu |
|---|---|
| Tính trung thực | `Number precision >= 98%` trên test set đã gán nhãn; 100% số sinh ra phải có trạng thái kiểm chứng |
| Chất lượng tóm tắt | Báo cáo ROUGE-1/2/L, BERTScore và đánh giá người; không chọn model chỉ bằng ROUGE |
| Định dạng đầu ra | 250-400 từ, đủ bốn mục cố định; nguyên nhân/rủi ro phải có evidence |
| Tái lập | Chạy lại cùng commit, data version, seed cho metric sai khác không quá ngưỡng đã công bố |
| Độ trễ | Ước lượng: xử lý một báo cáo dưới 120 giây trên GPU nhóm; chưa phải SLA production |
| Quy mô | Ước lượng giai đoạn đồ án: 1.000-3.000 báo cáo, 2-10 GB text/OCR, checkpoint 1-10 GB |
| Riêng tư | Chỉ dùng tài liệu công khai hoặc tài liệu nhóm có quyền sử dụng; không commit dữ liệu nhạy cảm |
| Chi phí/tính toán | Dùng Google Colab Pro; VRAM và thời lượng phiên không ổn định nên bắt buộc mixed precision, gradient accumulation, checkpoint và resume |
| Khả dụng | Batch/research workflow; job lỗi phải resume được từ checkpoint, không yêu cầu 24/7 |

Các con số trên là **mục tiêu khởi tạo**, phải được cập nhật sau tuần khảo sát dữ liệu và phần cứng.

### 2.4 Các quyết định đã chốt

| Hạng mục | Quyết định |
|---|---|
| Tài nguyên | Google Colab Pro; thiết kế job chịu được ngắt phiên và thay đổi loại GPU |
| Phạm vi doanh nghiệp | Corpus chính là doanh nghiệp phi tài chính thuộc VN30/VN100 trên HOSE; HNX là tập mở rộng tùy tiến độ vì VN30/VN100 là chỉ số của HOSE |
| Loại báo cáo | Báo cáo quý và năm; metadata phải phân biệt hợp nhất/riêng lẻ và không so sánh chéo sai loại |
| Nội dung nguồn | Ba báo cáo tài chính chính cùng giải trình biến động và thuyết minh trọng yếu |
| Đầu ra | Executive Summary tiếng Việt 250-400 từ theo đúng bốn mục đã định nghĩa |
| Mô hình sinh chính | Fine-tune BARTpho; ViT5 chỉ là baseline tùy chọn |
| Gán nhãn | Training dùng một annotator + một reviewer; validation/test dùng hai người độc lập |
| Xử lý bất đồng | TV4 là người phân xử dựa trên tài liệu gốc và annotation guideline |
| Pháp lý dữ liệu | Không làm legal review chuyên sâu cho đồ án; vẫn ghi nguồn/điều khoản, không công khai raw PDF hoặc dataset khi chưa có quyền phân phối |

## 3. Dữ liệu đề xuất

Không nên giả định có sẵn một dataset vừa chứa toàn văn BCTC Việt Nam, vừa có tóm tắt chuẩn, vừa có nhãn bằng chứng số. Dataset mục tiêu nhiều khả năng phải ghép từ báo cáo công khai và nhãn do nhóm xây dựng.

### 3.1 Nguồn ứng viên

| Nguồn | Có thể dùng cho | Hạn chế bắt buộc kiểm tra |
|---|---|---|
| [ViFinQA](https://huggingface.co/datasets/HuuDong03uet/ViFinQA) | 1.973 báo cáo OCR từ 100 công ty, giai đoạn 2015-2025; phù hợp tiền xử lý, truy xuất và kiểm tra số | Là QA, không có summary gold; cần ưu tiên doanh nghiệp phi tài chính VN30/VN100 và ghi lại điều khoản nguồn |
| [ViFinTab](https://huggingface.co/datasets/ngnquan/ViFinTab) | Phát hiện bảng trong BCTC, hỗ trợ giữ số liệu | Là object detection, không phải summarization; GPL-3.0 cần được TV5 kiểm tra tác động giấy phép |
| [Vietnamese Financial Summary](https://huggingface.co/datasets/Zenng2812/vietnamese-financial-summary) | Dữ liệu phụ để thử fine-tune tóm tắt và giữ số | Là tin tức tài chính, không phải BCTC; CC BY-NC 4.0, chỉ phù hợp mục đích phi thương mại và phải ghi nguồn |
| [OpenHust Vietnamese Summarization](https://huggingface.co/datasets/OpenHust/vietnamese-summarization) | Pre-fine-tune/baseline tóm tắt tiếng Việt | Dataset card thiếu thông tin; phải audit nguồn, license và chất lượng trước khi dùng |
| Công bố thông tin HOSE/HNX/UBCK và trang IR doanh nghiệp | Tài liệu BCTC gốc công khai | Phải tôn trọng điều khoản sử dụng, robots/rate limit và bản quyền; lưu URL, ngày tải, checksum |

Danh sách này là điểm bắt đầu. Vì đây là đồ án học thuật, nhóm không cần quy trình legal review phức tạp, nhưng TV1 vẫn phải lập `DATA_SOURCES.md` với URL, điều khoản/license nếu có, ngày tải và checksum. Không đưa raw PDF hoặc bản sao dataset lên GitHub công khai khi chưa rõ quyền phân phối.

### 3.2 Dataset mục tiêu tối thiểu

Mỗi mẫu nên có:

```json
{
  "document_id": "sha256:...",
  "company_id": "...",
  "exchange": "HOSE",
  "index_group": "VN30",
  "sector_type": "non_financial",
  "report_type": "quarterly_consolidated",
  "period_end": "2025-12-31",
  "source_url": "https://...",
  "pages": [{"page": 1, "text": "..."}],
  "sentences": [{"sentence_id": "p1-s1", "text": "...", "page": 1}],
  "facts": [{"fact_id": "f1", "metric": "revenue", "value": "1230000000", "unit": "VND", "period": "2025", "evidence_ids": ["p12-s3"]}],
  "gold_extractive_ids": ["p12-s3"],
  "gold_summary": {
    "business_results": "...",
    "financial_position_and_cash_flow": "...",
    "main_drivers": "...",
    "risks_and_watch_items": "..."
  },
  "annotation_status": "reviewed"
}
```

Quy tắc dữ liệu:

1. Chia train/validation/test theo `company_id`, không chia ngẫu nhiên theo chunk hay trang.
2. Không để các phiên bản cùng một báo cáo, báo cáo điều chỉnh hoặc bản mẹ/con gần trùng nằm ở nhiều split.
3. Dùng MinHash hoặc checksum sau chuẩn hóa để phát hiện trùng lặp.
4. Test set đóng băng trước khi tối ưu hyperparameter; người train model không sửa nhãn test tùy ý.
5. Mỗi summary test phải được hai người kiểm tra độc lập; TV4 phân xử bất đồng.
6. Giữ nguyên bản gốc bất biến; mọi bước làm sạch tạo phiên bản mới có lineage.
7. Số tiền được lưu dạng decimal/string chuẩn hóa, không dùng `float`.
8. Corpus chính chỉ giữ doanh nghiệp phi tài chính thuộc VN30/VN100 tại kỳ lấy mẫu; lưu snapshot danh sách chỉ số. Dữ liệu HNX nếu bổ sung phải mang nhãn `external_hnx` và không trộn vào test chính.
9. Không so sánh YoY giữa báo cáo hợp nhất và báo cáo riêng lẻ hoặc giữa kỳ có độ dài khác nhau.

Mục tiêu khả thi ban đầu: 1.000+ báo cáo không nhãn để khảo sát/tiền xử lý, 300-500 cặp có summary để train, 40-50 validation và 50-100 test. Trước tiên làm pilot 20 báo cáo. Nếu không đủ người gán nhãn, giảm số doanh nghiệp hoặc kỳ báo cáo thay vì sinh nhãn tự động rồi coi là ground truth.

### 3.3 Quy trình gán nhãn và xử lý bất đồng

Mỗi mẫu gán nhãn phải có câu/đoạn quan trọng cho extractor, summary đủ bốn mục, các claim số đã chuẩn hóa và `evidence_ids` trỏ về trang/câu/bảng nguồn.

**Training set:** một thành viên trong TV1-TV3 tạo nhãn, một thành viên khác review. Mẫu chỉ chuyển sang `reviewed` khi hai người thống nhất. Nếu không thống nhất, TV4 đọc nguồn và phân xử.

**Validation/test set:** hai thành viên trong TV1-TV3 làm độc lập, không xem kết quả của nhau. TV4 xử lý mọi khác biệt và đóng băng nhãn sau khi chốt. Người huấn luyện model không tự sửa test set dựa trên output của model.

TV4 viết `ANNOTATION_GUIDELINE.md` trước pilot, bao gồm:

1. Cách chọn câu quan trọng và giới hạn 250-400 từ.
2. Công thức YoY, biên lợi nhuận, quy tắc đơn vị và làm tròn.
3. Điều kiện một nguyên nhân được phép đưa vào summary.
4. Ngưỡng/tiêu chí chọn điểm lưu ý hoặc rủi ro.
5. Cách xử lý số liệu điều chỉnh, hợp nhất/riêng lẻ và bằng chứng mâu thuẫn.

TV5 kiểm tra ngẫu nhiên 10% mẫu về đủ field, evidence và trạng thái review. BARTpho hoặc LLM có thể tạo bản nháp để giảm công, nhưng bản nháp chỉ trở thành gold sau khi con người kiểm tra đầy đủ.

## 4. Hợp đồng giữa các module

### 4.1 Interface bắt buộc

| Module | Input | Output | Khi lỗi |
|---|---|---|---|
| `preprocess` | URI/file PDF + metadata | `DocumentRecordV1` | Trả mã lỗi `UNREADABLE`, không tạo text rỗng |
| `extractor` | `DocumentRecordV1` | `EvidenceBundleV1` | Báo `NO_EVIDENCE` nếu không đạt ngưỡng; không bịa câu |
| `generator` | `EvidenceBundleV1` | `SummaryDraftV1` | Timeout/OOM phải lưu config và trạng thái job để chạy lại |
| `verifier` | Draft + evidence + document | `VerifiedSummaryV1` | Claim không tìm thấy bằng chứng mang trạng thái `unsupported` |
| `pipeline` | Config + document URI | Artifact cuối | Exit code khác 0 nếu bước bắt buộc thất bại |

`EvidenceBundleV1` tối thiểu phải chứa `document_id`, `sentence_id`, `page`, `text`, `salience_score`, `source_order` và danh sách facts chuẩn hóa. Generator không được nhận một chuỗi mất toàn bộ metadata nguồn.

`SummaryDraftV1` phải chứa bốn field `business_results`, `financial_position_and_cash_flow`, `main_drivers`, `risks_and_watch_items`, tổng độ dài 250-400 từ và claim list kèm evidence. Validator từ chối output thiếu mục hoặc sai độ dài thay vì âm thầm coi là hợp lệ.

### 4.2 Quy tắc nhất quán và kiểm chứng số

1. Chuẩn hóa dấu phân cách Việt Nam, đơn vị (`đồng`, `nghìn`, `triệu`, `tỷ`), phần trăm, số âm trong ngoặc và kỳ báo cáo.
2. Một claim chỉ là `verified` khi giá trị, đơn vị, metric/entity và kỳ đều khớp bằng chứng.
3. Claim tính toán phải lưu phép toán và các fact đầu vào; dùng `Decimal` với quy tắc làm tròn công khai.
4. Nếu cùng một số xuất hiện ở nhiều chỗ, không coi khớp chuỗi là đủ; phải khớp ngữ cảnh.
5. Pipeline thử sinh lại tối đa một lần với feedback về claim lỗi. Nếu vẫn lỗi, kết quả là `REJECTED` hoặc loại câu lỗi theo cấu hình đánh giá; không âm thầm thay số.
6. Mọi kết quả cuối lưu danh sách claim, trạng thái và `evidence_ids` để audit.

### 4.3 Idempotency và phiên bản

Khóa một lần chạy là:

```text
run_id = hash(document_checksum, data_version, code_commit, model_version, config_hash)
```

Chạy lại cùng khóa không ghi đè artifact tốt bằng artifact dở. Schema dùng tên `DocumentRecordV1`, `EvidenceBundleV1`... Thay đổi phá vỡ tương thích phải tăng phiên bản và có migration hoặc tạo lại artifact.

## 5. Phân công 5 thành viên

Thay `TV1`...`TV5` bằng tên thật trong buổi kickoff. Tỷ trọng tương đối là `21% / 21% / 21% / 21% / 16%`. TV5 ít feature hơn nhưng có quyền và trách nhiệm tích hợp cao hơn.

### TV1 - Data Engineer, tiền xử lý và quản trị dataset (21%)

**Sở hữu:** `src/data/`, `src/preprocess/`, `configs/data/`, data card và phiên bản dữ liệu.

**Công việc:**

- Khảo sát ViFinQA, ViFinTab, nguồn sàn/doanh nghiệp; lập ma trận license và chất lượng.
- Viết downloader có rate limit, retry có giới hạn, checksum và provenance; không scrape khi điều khoản không cho phép.
- Trích xuất text/OCR, tách trang/câu, nhận diện bảng và chuẩn hóa Unicode nhưng giữ nguyên raw artifact.
- Xây schema `DocumentRecordV1`, kiểm tra trùng lặp và chia split theo công ty.
- Tạo script thống kê độ dài, tỷ lệ OCR lỗi, loại báo cáo, ngành và năm.
- Phối hợp TV4 thiết kế schema fact và bộ 100 mẫu kiểm tra thủ công.
- Bàn giao ngay 20 báo cáo pilot và fixture chuẩn cuối Tuần 1; không chờ hoàn thiện toàn bộ corpus.

**Đầu ra nghiệm thu:**

- `DATA_SOURCES.md`, `DATA_CARD.md`, manifest có checksum và script tái tạo dataset.
- Ít nhất 100 tài liệu pilot qua pipeline tiền xử lý; 95% record qua schema validation.
- Test cho số Việt Nam, trang mất text, file trùng, encoding và split leakage.

### TV2 - ML Engineer, BiLSTM-Attention train from scratch (21%)

**Sở hữu:** `src/models/extractor/`, `configs/extractor/`, metric trích xuất.

**Công việc:**

- Chốt đơn vị trích xuất là câu/đoạn và cách tạo nhãn extractive từ summary gold.
- Xây baseline đơn giản trước: lead-k và TF-IDF/TextRank.
- Huấn luyện embedding + BiLSTM + Attention từ đầu, có mask padding và seed cố định.
- Xuất top-k theo budget token, giữ thứ tự nguồn và salience score.
- Theo dõi Precision/Recall/F1, Recall@token-budget, coverage fact và attention sanity check.
- Bàn giao `EvidenceBundleV1` đúng contract; phối hợp TV3 về token budget.
- Phát triển từ fixture và nhãn pilot; dùng lead-k/TextRank làm fallback để TV3 không phải chờ model hoàn chỉnh.

**Đầu ra nghiệm thu:**

- Training/evaluation CLI, checkpoint manifest và báo cáo so với baseline.
- Unit test mask/shape/top-k/order; smoke train trên 20 mẫu; overfit được batch nhỏ.
- Không nhận model nếu không vượt ít nhất một baseline đã chốt hoặc không giải thích được nguyên nhân.

### TV3 - NLP Engineer, fine-tune BARTpho (21%)

**Sở hữu:** `src/models/generator/`, `configs/generator/`, inference và metric sinh.

**Công việc:**

- Xây baseline pretrained và pipeline fine-tune BARTpho làm generator chính.
- Chỉ chạy ViT5 như baseline tùy chọn nếu Colab Pro và tiến độ cho phép; không để baseline làm chậm BARTpho.
- Thiết kế input có marker câu/trang/fact, giới hạn token và prompt/task prefix nhất quán.
- Cấu hình decoding hạn chế lặp; lưu toàn bộ generation config.
- Đánh giá ROUGE, BERTScore, độ dài, repetition và số claim bị verifier từ chối.
- Hỗ trợ retry generation với feedback từ verifier, tối đa một lần.
- Trong khi chờ TV2, train/infer bằng gold evidence, fixture hoặc output lead-k/TextRank qua cùng `EvidenceBundleV1`.

**Đầu ra nghiệm thu:**

- Model card, script train/evaluate/infer và bảng ablation dùng/không dùng extractor.
- Test tokenizer, truncation, deterministic greedy decode và OOM-friendly smoke train.
- Checkpoint có thể resume sau khi Colab ngắt phiên; cấu hình mặc định dùng mixed precision và gradient accumulation.

### TV4 - Verification & Evaluation Engineer (21%)

**Sở hữu:** `src/verifier/`, `src/evaluation/`, annotation guideline và test set đóng băng.

**Công việc:**

- Định nghĩa claim số, entity/metric, kỳ, đơn vị, phép so sánh và phép tính.
- Xây bộ parser/normalizer dùng `Decimal`; liên kết claim với sentence/table evidence.
- Trả `verified`, `unsupported`, `conflict`, `ambiguous_unit` hoặc `wrong_period` cùng lý do.
- Viết guideline gán nhãn, phân xử bất đồng, tổ chức independent annotation cho validation/test và tính agreement phù hợp.
- Xây evaluation harness cho từng stage và end-to-end; thực hiện error analysis theo nhóm lỗi.
- Tổ chức đánh giá người về đầy đủ, dễ đọc, trung thực; ẩn tên model khi chấm.

**Đầu ra nghiệm thu:**

- Ít nhất 100 test case số có vàng cho dấu âm, %, đơn vị, kỳ, tăng/giảm và phép tính.
- Báo cáo number precision/recall, unsupported claim rate và confusion matrix theo loại lỗi.
- Verifier không crash khi không có số, OCR lỗi hoặc bằng chứng mâu thuẫn.

### TV5 - Maintainer, Integration & Release Owner (16%)

**Sở hữu:** kiến trúc repo, `src/pipeline/`, `schemas/`, CI, dependency lock, release và quyền merge `main`.

**Công việc:**

- Tạo cấu trúc package, contract/schema chung, config convention và CLI end-to-end tối thiểu.
- Tạo fixture V1, mock producer/consumer và pipeline mock trong Tuần 1 để các module phát triển song song.
- Thiết lập lint, format, type check, unit/integration test và branch protection.
- Review mọi PR về contract, test, độ phình dependency/artifact, bảo mật và khả năng tái lập.
- Chỉ merge khi CI xanh và đủ approval; quản lý release tag, changelog và rollback.
- Tích hợp module theo chiều dọc mỗi tuần; duy trì một golden sample chạy end-to-end.
- Theo dõi issue/risk/dependency, điều phối sửa contract và tránh hai người sửa cùng vùng.
- Kiểm tra secret, license và dữ liệu/checkpoint lớn trước khi merge.

**Đầu ra nghiệm thu:**

- Pipeline chạy bằng một lệnh trên sample nhỏ; CI dưới 10 phút, không chạy full training.
- `CONTRIBUTING.md`, PR template, CODEOWNERS/branch rule và release checklist.
- Báo cáo integration hàng tuần gồm commit, data version, model version, metric và lỗi mở.

TV5 không phải người sửa hộ mọi PR hoặc reviewer chuyên môn duy nhất. Tác giả chịu trách nhiệm đưa PR về trạng thái đạt chuẩn; owner module review nội dung chuyên môn, TV5 kiểm tra contract/CI trước khi merge. PR do TV5 tạo phải được ít nhất một thành viên có liên quan duyệt; quyền merge dự phòng giao cho TV4 khi TV5 vắng mặt hoặc có xung đột lợi ích.

## 6. Phối hợp, bàn giao và chống chờ

### 6.1 Cơ chế phát triển song song

Schema V1 và 10-20 fixture JSON phải được chốt trong Tuần 1. Mọi module cung cấp cả implementation thật và fixture/mock output đúng contract, nhờ đó module sau không phải chờ module trước hoàn thành.

| Thành viên | Đầu vào tạm thời được phép dùng | Fallback bắt buộc |
|---|---|---|
| TV1 | 20 báo cáo pilot tải thủ công | Text + bảng đã bóc tách sẵn nếu OCR chưa xong |
| TV2 | Fixture `DocumentRecordV1`, nhãn pilot | Lead-k và TF-IDF/TextRank luôn xuất được `EvidenceBundleV1` |
| TV3 | Gold evidence hoặc fixture `EvidenceBundleV1` | Lead-k/TextRank thay extractor thật |
| TV4 | Summary đúng/sai viết thủ công | Mutation đổi số, đơn vị, kỳ và dấu để test verifier |
| TV5 | Mock output của toàn bộ module | Tiny/golden pipeline chạy CPU trong CI |

Quy tắc chống chờ:

1. Không có task nào kéo dài quá ba ngày làm việc mà không tạo ra PR, fixture, báo cáo hoặc artifact có thể review.
2. Contract chỉ thay đổi qua PR riêng; không trộn thay đổi schema vào PR huấn luyện model.
3. Mỗi dependency phải có fallback baseline. Model chưa xong không được chặn pipeline.
4. PR phải có reviewer chuyên môn trong 24 giờ làm việc; quá hạn chuyển reviewer dự phòng.
5. Tích hợp end-to-end hằng tuần, không chờ đến khi tất cả model đạt metric mục tiêu.
6. TV3 duy trì hai chế độ input `gold` và `extractor`; TV4 duy trì bộ draft cố định độc lập với BARTpho.

### 6.2 Ma trận trách nhiệm

| Hạng mục | Responsible | Accountable | Consulted | Reviewer chính |
|---|---|---|---|---|
| Nguồn và split dữ liệu | TV1 | TV1 | TV2, TV4 | TV5 |
| Extractor | TV2 | TV2 | TV1, TV3 | TV5 |
| Generator | TV3 | TV3 | TV2, TV4 | TV5 |
| Verifier/evaluation | TV4 | TV4 | TV1, TV3 | TV5 |
| Annotation/adjudication | TV1-TV3 / TV4 phân xử | TV4 | TV5 audit 10% | TV5 |
| Contract và pipeline | TV5 | TV5 | TV1-TV4 | TV4 |
| Release cuối | TV5 | Cả nhóm | TV1-TV4 | TV4 + một owner liên quan |

Khi thay đổi schema, người đề xuất mở issue/ADR ngắn, cập nhật fixture contract và xin duyệt từ mọi owner bị ảnh hưởng trước khi sửa implementation.

## 7. Luật Git, review và kiểm thử bắt buộc

### 7.1 Luật Git

1. Không push trực tiếp vào `main`; mọi thay đổi qua PR.
2. Nhánh theo mẫu `feat/<module>-<issue>`, `fix/...`, `docs/...`, tồn tại ngắn và rebase/merge main thường xuyên.
3. Mỗi PR chỉ giải quyết một mục tiêu; khuyến nghị dưới 400 dòng code thay đổi, không tính lock/generated file. PR lớn phải có lý do và kế hoạch review theo phần.
4. Cấm commit raw dataset, PDF, checkpoint, tensorboard log, notebook output hoặc file trên 10 MB.
5. Dữ liệu/model dùng DVC hoặc object storage/Hugging Face nội bộ; Git chỉ lưu manifest, checksum, config và script tạo lại.
6. Không commit token, API key, cookie hoặc đường dẫn máy cá nhân. Secret dùng environment variable và file `.env` đã ignore.
7. Commit message mô tả thay đổi; không dùng commit kiểu `update`, `fix stuff`.
8. Reviewer phản hồi vòng đầu trong 24 giờ làm việc; TV4 là merge/review backup khi TV5 bị nghẽn.

### 7.2 Definition of Done cho mọi PR

Một PR chỉ được merge khi thỏa tất cả điều kiện sau:

- Có issue/mục tiêu và mô tả hành vi trước/sau.
- Có test mới hoặc giải thích hợp lý tại sao chỉ thay tài liệu.
- Unit test, schema test, lint, format và type check đều xanh.
- Nếu sửa contract, fixture consumer của module kế tiếp cũng được cập nhật.
- Không làm giảm metric golden set quá ngưỡng cho phép nếu chưa có quyết định được ghi lại.
- Không thêm dependency nếu standard library/dependency hiện tại giải quyết đủ; dependency mới phải nêu license và kích thước.
- Không chứa artifact lớn, secret, dữ liệu cá nhân hay nguồn dữ liệu chưa được duyệt.
- Có ít nhất một reviewer chuyên môn; TV5 kiểm tra cuối và merge.

### 7.3 Tháp kiểm thử

| Tầng | Khi chạy | Nội dung | Owner |
|---|---|---|---|
| Unit | Mỗi commit/PR | Parser, tokenizer wrapper, mask, shape, metric, số và đơn vị | Owner module |
| Schema/contract | Mỗi PR | Producer/consumer đọc cùng fixture V1 | TV5 + owner |
| Data validation | Mỗi data version | Null, duplicate, leakage, range, provenance, split | TV1 |
| Smoke training | Mỗi PR thay code model | 10-20 batch, loss hữu hạn, save/load checkpoint | TV2/TV3 |
| Integration | Mỗi PR vào main | Golden document chạy qua toàn pipeline bằng tiny model | TV5 |
| Regression | Hàng tuần và trước release | Fixed validation subset, so metric với baseline | TV4 |
| Full experiment | Theo milestone | Full training có log/artifact, không chạy trong CI | Owner model |
| Human evaluation | Trước báo cáo giữa kỳ/cuối kỳ | Blind review chất lượng và factuality | TV4 |

Ngưỡng khởi tạo cho CI:

- Coverage code thuần Python mới tối thiểu 80%; không dùng coverage để ép test dòng code framework vô nghĩa.
- Không có `NaN/Inf` trong smoke train; checkpoint save/load cho cùng greedy output trên fixture.
- Golden integration phải hoàn thành dưới 10 phút trên runner CPU hoặc tiny fixture.
- Summary validator phải pass các ca biên 250/400 từ và reject 249/401 từ hoặc thiếu một trong bốn mục.
- Number verifier phải đạt 100% trên bộ rule unit test và không giảm so với baseline trên golden set.
- Sai khác metric do seed phải nằm trong khoảng nhóm công bố từ ít nhất ba lần chạy ở thí nghiệm quan trọng.

### 7.4 Nhịp làm việc

- Mỗi ngày: tác giả chạy unit/smoke test liên quan trước khi push.
- Hai lần mỗi tuần: merge window do TV5 phụ trách; PR khẩn cấp vẫn phải đủ test/review.
- Trong tuần: owner module review chuyên môn, không dồn toàn bộ review cho TV5.
- Cuối mỗi tuần: chạy regression set, xem metric, kích thước artifact, thời gian chạy và issue tồn.
- Mỗi hai tuần: xóa code/config không dùng, hợp nhất config trùng và cập nhật risk register.
- Trước demo 48 giờ: đóng băng feature; chỉ nhận bug fix có test và phương án rollback.

## 8. Kế hoạch giao hàng theo giai đoạn

| Giai đoạn | Mục tiêu | Điều kiện qua cổng |
|---|---|---|
| Tuần 1 - Contract/pilot | Chốt nguồn, schema V1, annotation guideline, fixture, golden sample và pipeline mock | 20 báo cáo pilot; 10-20 fixture; mock end-to-end chạy được dù chưa có model thật |
| Tuần 2 - Baseline | Preprocess pilot, lead-k/TextRank, BARTpho baseline từ gold evidence, verifier trên draft cố định | Có báo cáo metric end-to-end đầu tiên và không module nào chờ module khác |
| Tuần 3-4 - Model | Train BiLSTM-Attention; fine-tune BARTpho; mở rộng verifier và annotation | Mỗi module vượt/so được baseline; checkpoint resume được trên Colab |
| Tuần 5 - Tích hợp thật | Thay dần mock bằng model thật, pipeline có evidence và retry/reject; data split đóng băng | Golden set chạy ổn định; không có contract mismatch |
| Tuần 6 - Đánh giá | Full experiment, ablation, human evaluation, error analysis | Có kết quả 3 seed cho thí nghiệm chính và factuality report |
| Tuần 7 - Ổn định | Sửa lỗi, tối ưu tài nguyên, tài liệu tái lập | Người mới chạy được sample từ README; release candidate |
| Tuần 8 - Báo cáo/demo | Đóng gói model card, data card, báo cáo và demo | Release tag, checksum artifact, rollback package và kết quả cuối |

Nếu thời gian ngắn hơn, bỏ baseline ViT5 và giảm số experiment phụ; vẫn giữ BARTpho, data audit, verifier, test split và integration test.

## 9. Vận hành, lỗi và rollback

### 9.1 Các lỗi dự kiến

| Lỗi | Hành vi bắt buộc |
|---|---|
| PDF/OCR không đọc được | Ghi lỗi theo `document_id`, không đưa record rỗng vào training |
| OOM khi train/infer | Giảm batch/token theo config đã định; resume checkpoint; không đổi ngầm config |
| Checkpoint hỏng | Kiểm checksum, quay lại checkpoint gần nhất và giữ log lỗi |
| Dữ liệu trùng/leak | Chặn tạo release dataset, tạo lại split và đánh dấu mọi metric cũ không hợp lệ |
| Generator sinh số sai | Retry tối đa một lần; sau đó reject/loại claim theo policy đã công bố |
| Schema không tương thích | Fail fast với version rõ ràng, không đoán field |
| Metric giảm | TV5 chặn merge nếu vượt regression budget; owner phân tích hoặc có ADR chấp nhận trade-off |

### 9.2 Quan sát và audit

Mỗi run lưu `run_id`, thời gian, commit, data/model/schema version, seed, config, phần cứng, metric, warning, số record lỗi và đường dẫn artifact. Không log toàn bộ tài liệu nếu chứa dữ liệu chưa được phép; log ID/hash và thống kê đủ để điều tra.

Mỗi checkpoint/release có checksum. Raw data là bất biến, processed data tái tạo được. Backup manifest/config/code trên Git; artifact lớn có ít nhất hai bản hoặc có thể tái tạo từ nguồn hợp lệ.

## 10. Các phương án không chọn

| Phương án | Lý do không chọn lúc này |
|---|---|
| Chỉ fine-tune BARTpho trực tiếp trên toàn báo cáo | Vượt context, khó truy vết, dễ mất bảng và số; không đáp ứng đề bài hai giai đoạn |
| Chỉ dùng extractive summary | Giữ factuality tốt hơn nhưng khó đọc và không đáp ứng phần sinh tóm tắt |
| Train cả ViT5 và BARTpho đầy đủ đến cuối | Gấp đôi compute/bảo trì trên Colab; BARTpho đã được chốt là model chính, ViT5 chỉ là baseline tùy chọn |
| Microservice cho từng model | Tăng CI/CD, network failure và vận hành mà đồ án batch chưa cần |
| Dùng LLM tự sinh toàn bộ gold summary | Nhanh nhưng tạo circular evaluation và hallucination; chỉ được dùng làm gợi ý, không là gold nếu chưa có người kiểm tra |
| Sửa số sai bằng cách tìm số gần nhất | Có thể tạo claim đúng số nhưng sai nghĩa/kỳ; verifier phải reject hoặc yêu cầu bằng chứng đầy đủ |

## 11. Tiêu chí hoàn thành đồ án

Đồ án chỉ được coi là hoàn thành khi:

1. Có dataset/data card hợp lệ, split không rò rỉ đã được kiểm tra.
2. Có baseline và kết quả riêng cho extractor, generator, verifier.
3. Có pipeline end-to-end tái lập từ config và artifact có phiên bản.
4. Mọi số trong output test có trạng thái và evidence; báo cáo cả trường hợp bị reject.
5. Có ablation chứng minh tác động của extractor và verifier.
6. Có đánh giá tự động lẫn đánh giá người, kèm error analysis chứ không chỉ một điểm ROUGE.
7. CI xanh, release được tag, tài liệu đủ để một thành viên khác chạy lại sample.
8. Báo cáo nêu rõ giới hạn dataset, OCR, generalization và việc hệ thống không phải tư vấn tài chính.

## 12. Rủi ro chưa giải quyết

| Rủi ro | Mức độ | Hướng giảm thiểu | Trigger đổi kế hoạch |
|---|---|---|---|
| Không có đủ summary gold | Cao | Training dùng annotator + reviewer; validation/test gán độc lập; giảm domain | Dưới 300 mẫu sạch: ưu tiên chất lượng và báo cáo giới hạn few-shot |
| Quyền phân phối dữ liệu không rõ | Trung bình | Lưu provenance, chỉ commit manifest/script, không public raw PDF | Có yêu cầu gỡ nguồn: loại artifact công khai và giữ script theo điều khoản |
| OCR bảng làm sai số | Cao | Giữ ảnh/trang, table detector, test số thủ công | Number recall quá thấp: giới hạn báo cáo có text layer |
| GPU không đủ | Trung bình | Model base/small, mixed precision, gradient accumulation | OOM kéo dài: giảm input budget, bỏ model so sánh |
| Summary đúng số nhưng sai ngữ cảnh | Cao | Verify metric/entity/period, không chỉ exact number | Precision dưới 98%: output dạng draft có cảnh báo, không gọi là verified |
| TV5 thành nút thắt merge | Trung bình | Owner review chuyên môn, SLA 24 giờ, checklist tự động, TV4 backup | PR chờ quá 24 giờ làm việc: chuyển reviewer/merge dự phòng |

Tài liệu này là baseline quản trị. Mọi thay đổi kiến trúc, schema, nguồn dữ liệu, metric chính hoặc tiêu chí factuality phải được ghi bằng ADR/issue và được các owner liên quan duyệt.
