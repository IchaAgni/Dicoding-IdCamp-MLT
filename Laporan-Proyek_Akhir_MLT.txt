# Laporan Proyek Machine Learning - Nisa Agni Afifah

## Project Overview

Dengan kemajuan teknologi, jumlah informasi yang tersedia semakin meningkat. Proses pencarian informasi melalui internet menjadi semakin sulit karena begitu banyaknya informasi yang tersedia. Jika pada masa lalu informasi hanya dapat diakses melalui media cetak, perkembangan teknologi telah menggeser penyediaan informasi ke media elektronik. Saat ini, dengan adanya internet, hampir semua jenis informasi sudah tersedia secara daring dalam berbagai versi, yang kadang membuat bingung karena jumlahnya yang begitu besar.

Trend yang sama terjadi dalam industri film. Menurut British Film Institute (BFI), jumlah film box office yang diproduksi terus meningkat dari tahun 2009 hingga 2015. Pada tahun 2009, ada 503 film yang diproduksi, sementara pada tahun 2015, jumlahnya meningkat menjadi 759 film. Karena jumlah film yang tersedia begitu banyak, sering kali penonton merasa kebingungan dalam memilih film yang ingin ditonton di internet[[Sistem Rekomendasi Film]](http://eprints.undip.ac.id/60611/1/laporan_24010312130054_1.pdf)

Oleh karena itu, diperlukan sebuah sistem yang dapat membantu menyaring informasi dan memberikan rekomendasi yang sesuai dengan preferensi pengguna. Sistem ini sering disebut sebagai sistem rekomendasi. Frank Kane, salah satu pionir Sistem Rekomendasi dalam course Building Recommender Systems with Machine Learning and AI, mendefinisikan sistem rekomendasi sebagai berikut.

*“A system that predicts ratings or preferences a user might give to a product. Often these are sorted and presented as “top-N'' recommendations. Also known as recommender engines, recommendation systems, and recommendation platforms”*. 

Sistem rekomendasi memprediksi rating atau preferensi pengguna terhadap item tertentu. Rekomendasi ini dibuat berdasarkan perilaku pengguna di masa lalu atau perilaku pengguna lainnya. Jadi, sistem ini akan merekomendasikan sesuatu terhadap pengguna berdasarkan data perilaku atau preferensi dari waktu ke waktu. 
Pada proyek ini untuk mendapatkan hasil rekomendasi menggunakan algoritma content based filtering dengan mencari kemiripan bobot dari term pada hasil pre-processing judul film dan rating film. Pembobotan dilakukan menggunakan metode TF-IDF yang telah dinormalisasi. Kemudian hasil pembobotan akan melalui tahap cosine similarity untuk mencari kemiripan berdasarkan bobot dan diakhiri dengan filtering berdasarkan genre. 

Referensi yang di gunakan :
- [Dicoding, Machine Learning Terapan ](https://www.dicoding.com/academies/319/corridor)
- [Sistem Rekomendasi](https://www.dicoding.com/academies/319/tutorials/17109)

## Business Understanding
Film merupakan salah satu jenis hiburan yang sering dikonsumsi oleh orang-orang untuk menghibur dirinya dari rutinitas melelahkan. Film sendiri memiliki definisi sebagai sebuah medium komunikasi audio visual yang tak hanya memberikan hiburan, tapi juga menawarkan informasi, dan bahkan bisa menyentuh emosi penontonnya. Menurut Hiawan Pratista (2008), film adalah media audio visual yang menggabungkan kedua unsur, yaitu naratif dan sinematik. Unsur naratif sendiri berhubungan dengan tema sedangkan unsur sinematik adalah alur atau jalan ceritanya yang runtun dari awal hingga akhir[[Definisi Film]](https://entertainment.kompas.com/read/2022/10/19/150302666/pengertian-film-definisi-jenis-dan-fungsinya?page=all)

Dalam proyek ini, salah satu faktor yang mempengaruhi minat seseorang untuk menonton film adalah genre film tersebut. Sebagai contoh, seseorang yang menyukai film Jhon Wick kemungkinan besar juga akan tertarik dengan film Nobody, karena keduanya memiliki genre yang sama, yaitu Action. Oleh karena itu, dibuatlah sistem rekomendasi menggunakan pendekatan Machine Learning untuk mendeteksi kemiripan dari suatu film yang telah ditonton dengan film-filmlainnya menggunakan data judulfilm tersebut maka dapat diurutkan berdasarkan genre film-film yang paling mirip dengan film yang telah ditonton dan akan dijadikan rekomendasi film yang akan ditonton selanjutnya. 

### Problem Statements
Berdasarkan latar belakang di atas, rincian masalahnya adalah sebagai berikut:
- Model *Machine Learning* apa yang cocok untuk menyelesaikan permasalahan tersebut?
- Bagaimana cara menentukan hasil rekomendasi suatu model *Machine Learning* yang dapat dikatakan baik?

### Goals
Untuk menjawab permasalahan di atas, maka akan goals/tujuan yang akan dicapai yaitu sebagai berikut:  
- Model yang cocok untuk menyelesaikan masalah tersebut adalah model yang berbasis dengan konten atau biasa disebut *Content-Based Filtering*.
- Melakukan evaluasi terhadap metrik dari model *Machine Learning* tersebut.
  
    ### Solution statements
    Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
* Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
    * Menangani missing value
    * Mengecek masalah data yang kosong dan membuang data yang kosong.
    * Menghapus data duplikat
    * mengurutkan data movieId dan menghapus data yg sama
* Metode yang digunakan pada projek ini adalah Content Based Filtering. Content Based Filtering adalah rekomendasi berbasis konten yang merekomendasikan item yang memiliki kemiripan dengan item yang disukai/diinput pengguna sebelumnya. Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna [[6](http://103.23.20.161/index.php/semnasif/article/view/1148)]. Metode ini bekerja dengan menyarankan item serupa yang pernah disukai sebelumnya atau sedang dilihat sekarang kepada pengguna berdasrakan kategori tertentu dari item yang dinilai oleh pengguna
 
## Data Understanding
Berikut merupakan informasi dari dataset yang digunakan:

|           Jenis         |  Keterangan |
| ----------------------- | ----------- |
|           Sumber        | [Movie Recommender System Dataset](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset)|
| Pemilik | [SHINIGAMI](https://www.kaggle.com/gargmanas) |
|          Lisensi        | [GPL 2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) |
| Jenis dan Ukuran Berkas | zip (846KB) |   

Tabel 1. Informasi Dataset  

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
