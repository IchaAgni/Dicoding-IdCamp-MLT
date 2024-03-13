# Laporan Proyek Machine Learning - Nisa Agni Afifah

## Domain Proyek

Topik proyek ini yaitu pada domain ekonomi dan bisnis, dengan berfokus pada rekrutmen karyawan baru oleh suatu perusahaan. Perusahaan ini tertarik untuk menilai kisaran gaji yang cocok berdasarkan pengalaman kerja calon pelamar yang akan direkrut.

### Latar Belakang
Sumber Daya Manusia (SDM) salah satu komponen yang berperan penting bagi
perusahaan, memegang peranan paling krusial dalam menentukan keberhasilan suatu tugas di dalam organisasi atau perusahaan. Dalam
mewujudkan tujuan yang optimal tentunya perusahaan
memerlukan tenaga suatu karyawan [Pengaruh Gaji Dan Insentif Terhadap Kinerja Karyawan](https://jurnal.stiekma.ac.id/index.php/JAMIN/article/view/53).

Dengan begitu perlu bagi perusahaan untuk memperhatikan para pekerjanya. Salah satu usaha untuk meningkatkan mutu SDM yaitu dengan pemberian gaji berdasarkan pengalaman kerja. Ketika seseorang yang sudah lama bekerja di suatu perusahaan maka gajinya akan semakin naik. Proyek ini ditujukan guna
menganalisis prediksi gaji karyawan berdasarkan lama tahun bekerja [Prediksi Gaji Berdasarkan Pengalaman Bekerja Menggunakan Metode Regresi Linear](https://doi.org/10.20895/dinda.v2i2.548) 

Dalam artikel berjudul “Competing on Analytics” yang diterbitkan oleh [Harvard Business Review](https://hbr.org/2006/01/competing-on-analytics), Davenport, seorang ahli *business analytics* berpendapat bahwa senjata strategis di bidang bisnis saat ini adalah pengambilan keputusan analitik. Ia merupakan teknik pengambilan keputusan berdasarkan berbagai informasi yang diekstrak dari data.

> Masalah ini diselasaikan dengan menghasilkan model yang dapat memprediksi kisaran gaji calon karyawan di perusahaan berdasarkan tahun pengalaman bekerja
Perusahaan akan mencoba menerapkan 2 model Machine Learning dan kemudian memilih model yang prediksinya paling mendekati. 

Referensi yang di gunakan :
- [Dicoding, Machine Learning Terapan ](https://www.dicoding.com/academies/319/corridor)
- [Predictive analytics](https://www.ibm.com/analytics/predictive-analytics)


## Business Understanding

Rekrutmen adalah proses mencari dan menyeleksi calon karyawan untuk mengisi posisi atau jabatan tertentu. Kunci utama menciptakan Sumber Daya Manusia (SDM) yang profesional terletak pada proses rekrutmen, seleksi, training, dan development calon karyawan. Tidaklah mudah mencari karyawan yang berkualitas. Proses rekrutmen ini penting dalam menentukan baik tidaknya pelamar yang melamar pekerjaan pada suatu perusahaan[Rekrutmen](https://majoo.id/solusi/detail/rekrutmen-adalah#:~:text=Rekrutmen%20adalah%20proses%20mencari%20dan,mudah%20mencari%20karyawan%20yang%20berkualitas)

Faktor yang dapat menjadi penentu efektivitas seorang karyawan adalah pengalaman kerja; semakin besar pengalamannya, semakin tinggi pula kualitasnya. Dikutip dari salah satu Jurnal Ekonomi Pembangunan, Manajemen dan Bisnis, Akuntansi “Pengalaman kerja karyawan berhubungan erat dengan kinerja”.Pengalaman kerja akan membantu kelancaran didalam menyelesaikan pekerjaan dalam suatu perusahaan[Dampak Masa Kerja, Pengalaman Kerja, Kemampuan Kerja Terhadap Kinerja Karyawan](https://e-journal.upr.ac.id/index.php/jemba/article/download/2986/2501)

Dalam proyek ini, perusahaan akan mengembangkan beberapa model Machine Learning yang nantinya dievaluasi untuk membandingkan tingkat akurasi prediksi. Harapannya, model-model tersebut mampu memprediksi kisaran gaji yang sesuai berdasarkan pengalaman kerja calon pelamar.

### Problem Statements
Berdasarkan latar belakang yang telah diuraikan, perincian masalahnya dapat dijabarkan sebagai berikut:
- Algoritma mana yang tepat untuk mengantisipasi kisaran gaji karyawan?
- Bagaimana cara menetapkan kriteria untuk menilai keberhasilan hasil prediksi dari suatu Algoritma Machine Learning?

### Goals
Untuk menjawab pertanyaan tersebut, dibuat "Predictive Modelling" dengan tujuan atau goals sebagai berikut :
- Meskipun tersedia banyak algoritma yang bisa menyelesaikan permasalahan tersebut, di proyek ini akan memilih menggunakan algoritma LinearRegression dan RandomForest.
- Melakukan evaluasi terhadap metrik dari masing-masing algoritma.

### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

