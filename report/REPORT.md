# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`=cab; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về cả bức ảnh vì task là image_classification và model là yolo11n-cls, tức là phân loại cả ảnh chứ không phát hiện vật thể. Không có bbox hay tọa độ nào, chỉ có kích thước của toàn ảnh.
3. Danh sách tên lớp đi kèm taxonomy/checkpoint, model không tự nghĩ ra. Model chỉ xuất 1000 điểm số. Vị trí cao nhất là id 468, rồi tra id đó trong danh sách ImageNet-1K được lưu trong file .pt thì ra “cab”.
4. Vì sao cần cả `class_id`, `class_name` và `taxonomy_name`:
   - **Chỉ có tên** thì dễ mơ hồ. Trong ImageNet có tên bị trùng, ví dụ hai lớp cùng tên “crane”: một là con sếu, một là cần cẩu. Tên cũng có thể viết khác nhau (“cab”, “taxi”, “taxicab”).
   - **Chỉ có id** thì vô nghĩa nếu không biết bảng nào. Số 468 ở ImageNet là “cab”, nhưng ở COCO hay một taxonomy khác lại là lớp khác hoặc không tồn tại.
   - **`taxonomy_name`** cho biết bộ nhãn nào đang được dùng. Nhờ vậy kết quả mới so sánh và tái lập được.
5. Guideline cần nói rõ:
   - **Tiêu chí chọn nhãn:** chủ thể chính, vật lớn nhất, vật ở trung tâm hay vật nổi bật nhất.
   - **Thứ tự ưu tiên** khi có nhiều vật ngang nhau.
   - **Mức độ chi tiết của nhãn:** chọn “cab” hay chọn nhãn chung hơn.
   - **Xử lý trường hợp khó:** làm gì khi không có lớp phù hợp hoặc ảnh quá mơ hồ (bỏ qua, gắn cờ, hay chọn “khác”).
   - **Ví dụ minh họa** cho các ca dễ gây tranh cãi.
6. Vì sao không dùng `score` làm đáp án đúng:
   - **Score là độ tự tin của model, không phải sự thật.** Model có thể sai mà vẫn rất tự tin, và score thường không được hiệu chỉnh (calibrate).
   - **Bị vòng lặp.** Nếu dùng dự đoán của model làm đáp án rồi chấm chính model đó thì kết quả lúc nào cũng “đúng”, và các lỗi của model bị giữ lại.
   - **Ở đây 0.51 khá thấp.** Gần một nửa xác suất thuộc về các lớp khác.
   - **Ground truth** phải do người gán nhãn quyết định theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bowl; `score`=0.701282; `bbox_xyxy`=[155.02, 168.67, 182.38, 184.18] ; `bbox_width`=27.36; `bbox_height`=15.51
2. Đọc bbox:
   - **Góc trên-trái** cách mép trái khoảng 155 px và cách mép trên khoảng 169 px.
   - **Góc dưới-phải** cách mép trái khoảng 182 px và cách mép trên khoảng 184 px.
   - **Kích thước:** box rộng khoảng 27 px, cao khoảng 16 px. Đây là một hộp nhỏ, dẹt.
   - **Trên hình,** đây là cái tô “bowl 0.70” đặt trên kệ phía trên bếp lò, hơi lệch về bên trái.
3. Trên ảnh kitchen, ở ngưỡng 0.35 có 11 prediction (2 person, 2 oven, 5 bowl, 2 cup), còn ở ngưỡng 0.60 chỉ còn 6 vì 5 hộp bowl và cup có score dưới 0.60 đã bị loại.
4. Khi hạ ngưỡng, reviewer phải làm thêm:
   - **Kiểm tra nhiều hộp hơn** và xóa các false positive, tức là hộp nhận sai hoặc hộp không có vật.
   - **Gộp hoặc xóa các hộp trùng,** chồng lên nhau trên cùng một vật. Ví dụ trong ảnh có hai nhãn “cup” đè lên nhau.
   - **Sửa nhãn sai lớp** và chỉnh lại các hộp lệch.
5. Mỗi vật thể có đúng một box. Box ôm sát phần nhìn thấy được của vật: bốn cạnh chạm vào các pixel ngoài cùng của vật, không chừa nền thừa, không tính bóng đổ hay phản chiếu, và không tự đoán phần bị che.
6. Vật bị cắt mép: “person 0.61” ở mép trái ảnh. Chỉ thấy được cánh tay và bàn tay, phần còn lại của người nằm ngoài khung.
   - **Phần chung** thì viết thành quy tắc trong guideline. Vật bị cắt ở mép ảnh xuất hiện ở rất nhiều ảnh, nên cần một cách xử lý thống nhất: vẫn gán nhãn nếu nhận ra được vật, và box chỉ ôm phần nhìn thấy, kéo đến sát mép ảnh.
   - **Phần ranh giới** thì đưa lên hỏi Lab Coach. Câu hỏi “chỉ thấy một bàn tay thì có tính là person không?” là quyết định chính sách về mức tối thiểu phải nhìn thấy, người gán nhãn không nên tự quyết. Sau khi Lab Coach trả lời thì bổ sung câu trả lời đó vào guideline để lần sau không phải hỏi lại.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001; `class_name`=person; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348; 3–5 điểm đầu của `polygon_xy`=[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]
2. Polygon cho biết gì mà box không cho biết:
   - **Box** chỉ có 4 số [385.45, 66.44, 498.02, 348.58], tức là một hình chữ nhật bao ngoài người.
   - **Polygon** có 348 điểm đi theo đúng đường viền của người: đầu, vai, khuỷu tay chống hông và hai chân.
   - **Nhờ vậy** biết được pixel nào thuộc người, pixel nào là nền. Ví dụ khoảng trống giữa tay và thân, hay phần tường lọt vào box, đều không thuộc người.
3. `instance_id` là gì:
   - **`instance_id`** là mã định danh của từng vật thể riêng lẻ trong ảnh.
   - **Không phải class ID.** `class_id` cho biết vật thuộc loại gì. Ví dụ kitchen-002, kitchen-003, kitchen-006 và kitchen-010 đều có cùng `class_id` 45 (bowl), nhưng là 4 cái tô khác nhau nên có 4 `instance_id` khác nhau.
   - **Không phải tracking ID.** ID này chỉ có giá trị trong một ảnh. Nó không theo dõi cùng một vật qua nhiều ảnh hay nhiều frame video. Ví dụ “kitchen-001” ở ảnh khác không phải là cùng một người.
4. Đi đường biên theo mép ngoài cùng của phần nhìn thấy của vật. Không lấy bóng đổ, phản chiếu hay nền lọt qua các khoảng hở. Không tự vẽ phần bị che. Mỗi pixel chỉ thuộc một instance; khi hai vật chồng nhau thì pixel thuộc về vật nằm phía trước.
5. Hai vật chạm nhau: các tô ở góc dưới-trái nằm trên bàn.
   - **Trường hợp:** bowl 0.74, 0.53 và 0.70, chạm vào nhau và chồng lên mask dining table.
   - **Dùng guideline.** Vật đặt chồng lên vật khác, ví dụ tô trên bàn, xuất hiện ở rất nhiều ảnh. Quy tắc “pixel thuộc vật phía trước” ở câu 4 đã xử lý được trường hợp này: mask của bàn phải chừa trống chỗ các tô.
   - **Chỉ escalation** khi guideline không phân định được. Ví dụ: không nhìn ra biên giữa hai tô chồng lên nhau, hoặc không chắc một vật có phải là bowl không (như cái khuôn bánh trắng). Khi đó mới hỏi Lab Coach, rồi bổ sung câu trả lời vào guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: class_id + class_name + taxonomy_name (ví dụ ImageNet-1K). | Ảnh traffic có rank 1 là cab, nhưng score chỉ 0.51. Kết quả detection của chính ảnh này lại có nhiều bus và car (bus 0.91, car 0.86, …). Ảnh có nhiều chủ đề nên khó chọn một nhãn. | Chọn một lớp trong taxonomy theo guideline (chủ thể chính / nổi bật nhất). Gắn cờ những ảnh mơ hồ. |Nhãn có đúng guideline không, các annotator có chọn nhất quán giữa các ảnh tương tự không, và các ảnh bị gắn cờ đã được xử lý chưa.  |
| Phát hiện vật thể | Mỗi vật thể một record: lớp (COCO-80) + bbox_xyxy tính bằng pixel. | kitchen: hộp person 0.61 [0.12, 263.18, 61.55, 310.78] chỉ có cánh tay bị cắt ở mép ảnh. Hai hộp cup 0.45 và 0.38 nằm sát nhau, có thể trùng. Hộp bowl 0.38 chồng lên bowl 0.72. Các xoong chảo treo trên tường không được nhận vì không có lớp phù hợp trong COCO-80. | Vẽ box ôm sát phần nhìn thấy, mỗi vật một box. Chọn lớp đúng, bỏ qua vật nằm ngoài taxonomy. | Có vật bị bỏ sót không, có hộp sai (FP) hay trùng không, box có ôm sát không, lớp có đúng không. Hạ ngưỡng thì số hộp cần kiểm tăng lên: 6 hộp ở 0.60 so với 11 hộp ở 0.35. |
| Instance segmentation |Mỗi vật thể một record: instance_id + lớp + polygon/mask (polygon_xy, tính bằng pixel).  | kitchen-005 (dining table, 0.58): mask tràn xuống cái xô dưới gầm bàn và chồng lên các tô. kitchen-007 (oven, 0.49): mask lẫn vào vùng bàn và các tô. kitchen-010 (bowl): chỉ có 12 điểm polygon nên biên rất thô. | Vẽ biên theo mép phần nhìn thấy. Mỗi pixel chỉ thuộc một instance, vật phía trước được giữ pixel. Gán instance_id riêng cho từng vật. | Biên có khớp vật không, có bị tràn hay hụt không, các mask có chồng chéo không, có instance bị bỏ sót không, lớp và instance_id có đúng không. |

## 5. An toàn dữ liệu

1. Chỉ dùng ảnh COCO và các file output trong day1_lab_outputs/ mà bài lab cung cấp; không tải lên, chia sẻ hay sao chép dữ liệu ra ngoài môi trường lab, và không thêm ảnh hay thông tin cá nhân nào khác.
2. Dừng ngay, không gán nhãn, không chỉnh sửa hay chia sẻ dữ liệu đó, rồi báo cho Lab Coach kèm sample_id hoặc tên file và mô tả ngắn vấn đề.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
