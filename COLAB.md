# Google Colab дээр CUDA-тай хувилбар ажиллуулах заавар

Энэхүү заавар нь `serial / std::thread / OpenMP / CUDA` 4 хувилбарыг
Google Colab-ийн T4 GPU дээр нэг ажиллуулаас хэмжиж, ижил `csv/output.csv`
үүсгэн графикуудыг гаргаж авах процессыг тайлбарлана.

## Алхам 1 — GPU runtime идэвхжүүлэх

Colab notebook нээгээд:

`Runtime → Change runtime type → Hardware accelerator: T4 GPU`

GPU байгаа эсэхийг шалгана:

```python
!nvidia-smi
```

## Алхам 2 — Эх кодыг авах

Repo-г clone хийнэ:

```python
!git clone https://github.com/Xvclnn/F.CSM306-project.git
%cd F.CSM306-project
```

## Алхам 3 — Compile хийх

`USE_CUDA` макроыг тодорхойлж, CPU + GPU 4 хувилбарыг нэг binary болгоно:

```python
!nvcc -O3 -std=c++17 -DUSE_CUDA -Xcompiler -fopenmp \
      src/main.cpp src/tasksys.cpp src/cuda.cu \
      -o src/main -lgomp
```

> Алдаа гарвал: `apt-get install -y libomp-dev` (OpenMP сан дутуу үед)

## Алхам 4 — Benchmark ажиллуулах

```python
!mkdir -p csv output
!cd src && ./main
```

Ажиллах үед stdout-д Serial / std::thread / OpenMP / CUDA 4 аргын хугацаа
харагдана. CUDA мөрөнд транзакцын хугацаа ба байт хэмжээ нэмэгдсэн байна.

## Алхам 5 — График үүсгэх

```python
!pip install pandas matplotlib -q
!python3 visualization/main.py
```

`output/` хавтсанд:
- `execution_time.png`
- `speedup.png`
- `achievable_performance.png`

3 файл бүгдэд нь CUDA шар өнгөтэй шугам нэмэгдсэн байна.

## Алхам 6 — Графикуудыг үзэх

```python
from IPython.display import Image, display
display(Image("output/execution_time.png"))
display(Image("output/speedup.png"))
display(Image("output/achievable_performance.png"))
```

## Алхам 7 — CSV-г татаж авах

```python
from google.colab import files
files.download("csv/output.csv")
files.download("output/execution_time.png")
files.download("output/speedup.png")
files.download("output/achievable_performance.png")
```

---

## CSV формат — нэмэлт CUDA талбарууд

CPU мөрүүдтэй яг адил формат хэвээр, гэвч CUDA мөрөнд `data_transfer_time`
ба `data_transferred_bytes` талбарууд тэг биш утгаар бичигдэнэ.

| method | input_size | num_threads | run_id | execution_time_ms | data_transfer_time | data_transferred_bytes | total_operations | achievable_performance |
|--------|-----------|-------------|--------|-------------------|--------------------|-----------------------|------------------|------------------------|
| serial | 1000000 | 1 | 1 | 92.5 | 0 | 0 | 40000000 | 4.32e8 |
| threads | 1000000 | 20 | 1 | 13.2 | 0 | 0 | 40000000 | 3.03e9 |
| openmp | 1000000 | 20 | 1 | 11.4 | 0 | 0 | 40000000 | 3.50e9 |
| **cuda** | 1000000 | 256 | 1 | 8.1 | 4.5 | 8000000 | 40000000 | 4.94e9 |

CUDA-ийн `num_threads` талбарт block_size (256)-ийг хадгална. Графикийн
бүтэц өөрчлөгдөхгүй – зөвхөн CUDA шугам нэмэгдэнэ.

---

## CPU дээр (CUDA-гүй) compile хийх

CUDA байхгүй машин дээр өмнөх компиляц хэвээр ажиллана:

```bash
g++ -std=c++17 -O2 -fopenmp src/main.cpp src/tasksys.cpp -o src/main
```

`USE_CUDA` тодорхойлогдоогүй учир CUDA блок хөрвөгдөхгүй – CSV дотор
`cuda` мөр гарахгүй. Графикт CUDA шугам байхгүй.
