# SFC-MAPPING PROBLEM EXPLAINED - Simulation

Mô phỏng bài toán ánh xạ mạng dịch vụ lên mạng vật lí.

## Bài toán
Cho đồ thị mạng vật lí $\mathcal{G}\left(\mathcal{N},\mathcal{E}\right)$ và tập hợp $\mathcal{S}$ là tập hợp gồm các mạng dịch vụ (SFC) được biểu diễn dưới dạng các đồ thị $\mathcal{G}_s\left(\mathcal{N}_s,\mathcal{E}_s\right)$.

Mạng vật lí có tập $\mathcal{N}$ chứa các nút mạng $i$ và tập $\mathcal{E}$ chứa các liên kết $ij$ giữa các nút mạng với nhau. 
Mỗi nút mạng sẽ có một lượng tài nguyên có sẵn là $a_{i}$, 
mỗi liên kết cũng sẽ có một giới hạn là $a_{ij}$.

Mỗi mạng dịch vụ có tập $\mathcal{N}_s$ chứa các nút là các VNF $v$ và tập $\mathcal{E}_s$ chứa các liên kết $vw$ giữa các nút với nhau. 
Mỗi một VNF sẽ cần một lượng tài nguyên nhất định để hoạt động là $r_{v}$, 
mỗi một liên kết giữa hai VNF bất kì cũng sẽ có những yêu cầu đặc thù về truyền dẫn, kí hiệu là $r_{vw}$.

Yêu cầu đặt ra đó là làm sao để ánh xạ toàn bộ các đồ thị $\mathcal{G}_{s}$ trong tập $\mathcal{S}$ vào đồ thị mạng vật lí $\mathcal{G}$.

## Về repo này

Repo này chứa mã nguồn để mô phỏng quá trình giải quyết bài toán nêu trên. Chi tiết về phương pháp và cơ sở lí luận của bài toán đã được giải thích chi tiết trong tài liệu [này]().

## Mô tả cấu trúc repo

Thư mục `.\mapping-solo` chứa mã nguồn và hướng dẫn thực hiện mô phỏng với bài toán ánh xạ một SFC lên mạng vật lí.

Thư mục `.\mapping-combine-simple` chứa mã nguồn và hướng dẫn thực hiện mô phỏng bài toán ánh xạ nhiều SFC trong tập $\mathcal{S}$ lên mạng vật lí, với kết quả hoặc chấp nhập ánh xạ toàn bộ, hoặc từ chối toán bộ SFC.

Thư mục `.\mapping-combine-maxima` chứa mã nguồn và hướng dẫn thực hiện mô phỏng bài toán ánh xạ nhiều SFC trong tập $\mathcal{S}$ lên mạng vật lí, tuy nhiên nếu mạng vật lí không đủ tài nguyên để ánh xạ toàn bộ thì sẽ thực hiện ánh xạ nhiều SFC nhất có thể.

Thư mục `.\test-cases-generator` chứa mã nguồn và hướng dẫn thực hiện sinh các bộ mẫu thử phục vụ cho việc thử nghiệm quá trình ánh xạ trong các bài toán nêu trên.

Thư mục `.\combo-simulation` chứa mã nguồn và hướng dẫn thực hiện so sánh giữa 3 phương pháp mô phỏng.