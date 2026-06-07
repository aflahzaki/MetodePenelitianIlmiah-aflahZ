# Evaluasi Performa Prediksi Probabilistik pada Klasifikasi Kualitas Air Menggunakan Algoritma Natural Gradient Boosting (NGBoost)

## Informasi Umum

| Item | Detail |
|------|--------|
| **Penulis** | Aflah Zaki Siregar |
| **NIM** | 103062300095 |
| **Mata Kuliah** | Metode Penelitian Ilmiah |
| **Tahun** | 2026 |
| **Template** | MDPI Journal |

## Deskripsi

Penelitian ini mengevaluasi performa algoritma Natural Gradient Boosting (NGBoost) dalam melakukan prediksi probabilistik untuk klasifikasi kualitas air. NGBoost merupakan metode ensemble learning yang mampu menghasilkan distribusi probabilitas penuh sebagai output prediksi, sehingga memberikan informasi ketidakpastian yang lebih kaya dibandingkan metode klasifikasi konvensional. Evaluasi dilakukan menggunakan metrik-metrik standar seperti akurasi, confusion matrix, feature importance, serta analisis distribusi probabilitas prediksi.

## Struktur Repositori

```
.
├── README.md
└── AflahZakiSiregar_103062300095_MetodePenelitianIlmiah_..._2026/
    ├── asi-3473165.tex          # File utama paper (LaTeX)
    ├── asi-3473165.pdf          # Hasil kompilasi PDF
    ├── sumber.bib               # Daftar pustaka (BibTeX)
    ├── Definitions/             # Template MDPI (class, style, logo)
    │   ├── mdpi.cls             # Class file MDPI
    │   ├── mdpi.bst             # Bibliography style MDPI
    │   ├── package.tex          # Package definitions
    │   └── ...                  # Logo dan file pendukung template
    └── image_rev/               # Gambar dan visualisasi
        ├── confusion_matrix_compare.png
        ├── diagram_flowchart.png
        ├── feature_importance.png
        ├── gambar_hasil.png
        ├── kde_prob_distribution.png
        └── loss_curve.png
```

## Cara Kompilasi

Untuk mengompilasi paper dari source LaTeX:

```bash
cd AflahZakiSiregar_103062300095_MetodePenelitianIlmiah_EvaluasiPerformaPrediksiProbabilistikpadaKlasifikasiKualitasAirMenggunakanAlgoritma_2026

# Kompilasi 3 pass (diperlukan untuk referensi dan bibliografi)
pdflatex -interaction=nonstopmode asi-3473165.tex
bibtex asi-3473165
pdflatex -interaction=nonstopmode asi-3473165.tex
pdflatex -interaction=nonstopmode asi-3473165.tex
```

Hasil kompilasi berupa file `asi-3473165.pdf`.

## Dependensi

Diperlukan instalasi **TeX Live** (atau distribusi LaTeX lainnya) dengan paket-paket berikut:

- `pdflatex` (compiler)
- `bibtex` (bibliography processor)
- `babel` (dukungan bahasa Indonesia)
- `natbib` (citation management)
- `tikz` / `pgfplots` (diagram dan grafik)
- `graphicx` (penyisipan gambar)
- `amsmath`, `amssymb` (notasi matematika)
- `hyperref` (hyperlink dalam PDF)
- `booktabs`, `tabularx` (tabel)
- `algorithm2e` (pseudocode)

Pada sistem berbasis Debian/Ubuntu:

```bash
sudo apt-get install texlive-full
# atau minimal:
sudo apt-get install texlive-latex-extra texlive-science texlive-lang-other texlive-bibtex-extra
```

## Lisensi

Repositori ini digunakan untuk keperluan akademik pada mata kuliah Metode Penelitian Ilmiah di Telkom University.
