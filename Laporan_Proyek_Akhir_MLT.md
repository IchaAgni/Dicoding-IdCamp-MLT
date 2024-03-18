# Laporan Proyek Machine Learning - Nisa Agni Afifah

## Project Overview

Dengan kemajuan teknologi, jumlah informasi yang tersedia semakin meningkat. Proses pencarian informasi melalui internet menjadi semakin sulit karena begitu banyaknya informasi yang tersedia. Jika pada masa lalu informasi hanya dapat diakses melalui media cetak, perkembangan teknologi telah menggeser penyediaan informasi ke media elektronik. Saat ini, dengan adanya internet, hampir semua jenis informasi sudah tersedia secara daring dalam berbagai versi, yang kadang membuat bingung karena jumlahnya yang begitu besar.

Trend yang sama terjadi dalam industri film. Menurut British Film Institute (BFI), jumlah film box office yang diproduksi terus meningkat dari tahun 2009 hingga 2015. Pada tahun 2009, ada 503 film yang diproduksi, sementara pada tahun 2015, jumlahnya meningkat menjadi 759 film. Karena jumlah film yang tersedia begitu banyak, sering kali penonton merasa kebingungan dalam memilih film yang ingin ditonton di internet[[1]](http://eprints.undip.ac.id/60611/1/laporan_24010312130054_1.pdf).

Oleh karena itu, diperlukan sebuah sistem yang dapat membantu menyaring informasi dan memberikan rekomendasi yang sesuai dengan preferensi pengguna. Sistem ini sering disebut sebagai sistem rekomendasi. Frank Kane, salah satu pionir Sistem Rekomendasi dalam course Building Recommender Systems with Machine Learning and AI, mendefinisikan sistem rekomendasi sebagai berikut.

*“A system that predicts ratings or preferences a user might give to a product. Often these are sorted and presented as “top-N'' recommendations. Also known as recommender engines, recommendation systems, and recommendation platforms”* [[2]](https://www.dicoding.com/academies/319/corridor).

Sistem rekomendasi memprediksi rating atau preferensi pengguna terhadap item tertentu. Rekomendasi ini dibuat berdasarkan perilaku pengguna di masa lalu atau perilaku pengguna lainnya. Jadi, sistem ini akan merekomendasikan sesuatu terhadap pengguna berdasarkan data perilaku atau preferensi dari waktu ke waktu[[3]](https://www.dicoding.com/academies/319/tutorials/17109).
Pada proyek ini untuk mendapatkan hasil rekomendasi menggunakan algoritma content based filtering dengan mencari kemiripan bobot dari term pada hasil pre-processing judul film dan rating film. Pembobotan dilakukan menggunakan metode TF-IDF yang telah dinormalisasi. Kemudian hasil pembobotan akan melalui tahap cosine similarity untuk mencari kemiripan berdasarkan bobot dan diakhiri dengan filtering berdasarkan genre. 

## Business Understanding
Film merupakan salah satu jenis hiburan yang sering dikonsumsi oleh orang-orang untuk menghibur dirinya dari rutinitas melelahkan. Film sendiri memiliki definisi sebagai sebuah medium komunikasi audio visual yang tak hanya memberikan hiburan, tapi juga menawarkan informasi, dan bahkan bisa menyentuh emosi penontonnya. Menurut Hiawan Pratista (2008), film adalah media audio visual yang menggabungkan kedua unsur, yaitu naratif dan sinematik. Unsur naratif sendiri berhubungan dengan tema sedangkan unsur sinematik adalah alur atau jalan ceritanya yang runtun dari awal hingga akhir[[4]](https://entertainment.kompas.com/read/2022/10/19/150302666/pengertian-film-definisi-jenis-dan-fungsinya?page=all).

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
    * Mengurutkan data movieId dan menghapus data yg sama
    * Mengonversi data series menjadi bentuk list
* Metode yang digunakan pada projek ini adalah Content Based Filtering. Content Based Filtering adalah Content-based filtering adalah metode yang digunakan dalam sistem rekomendasi dan analisis data yang berfokus pada karakteristik atau konten dari item-item yang ingin direkomendasikan atau dianalisis. Pendekatan ini menggunakan atribut-atribut atau fitur-fitur item untuk menentukan kesamaan antara item yang ada dan preferensi pengguna.Dalam konteks rekomendasi, content-based filtering berusaha untuk merekomendasikan item yang mirip dengan item yang telah disukai oleh pengguna berdasarkan karakteristik konten.

 ![image](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/img/cbf.png?raw=true)
 
## Data Understanding
Berikut merupakan informasi dari dataset yang digunakan:

|           Jenis         |  Keterangan |
| ----------------------- | ----------- |
|           Sumber        | [Movie Recommender System Dataset](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset)|
| Pemilik | [SHINIGAMI](https://www.kaggle.com/gargmanas) |
|          Lisensi        | [GPL 2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) |
| Jenis dan Ukuran Berkas | zip (846KB) |   

Tabel 1. Informasi Dataset  
Pada berkas tersebut terdapat 2 file, yaitu movies.csv dan ratings.csv

### Deskripsi Variabel

 - **movies.csv**

|  # |  Column  |  Non-Null Count |  Dtype |
|----|----------|-----------------|--------|
| 0  |  movieId | 2731 non-null   | int64  | 
| 1  |  tittle  | 2731 non-null   | object | 
| 2  |  genres  | 2731 non-null   | object | 

Variabel-variabel yang terdapat pada file movies.csv adalah sebagai berikut:

 - *movieId*: id film
 - *title*: Judul film
 - *genres*: genre film  

 - **ratings.csv**
   
|  # |  Column   |  Non-Null Count |  Dtype  |
|----|-----------|-----------------|---------|
| 0  |   userId  | 2731 non-null   | int64   | 
| 1  |  movieId  | 2731 non-null   | int64   | 
| 2  |  rating   | 2731 non-null   | float64 |
| 2  | timestamp | 2731 non-null   | int64   |

 Variabel-variabel yang terdapat pada file ratings.csv adalah sebagai berikut:  
   - *userId*: id user
   - *movieId*: id film
   - *rating*: rating yang diberikan user
   - *timestamp*: waktu user memberikan rating

  ### Exploratory Data Analysis - Univariate Analysis

 - **Movies**  
 ![unv-2](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/img/EDA-Movies.png?raw=true)  
 Gambar 3. Distribusi fitur genre

 Dari hasil visualisasi pada Gambar 3 dapat disimpulkan bahwa:

 Sebagian besar sampel film dari dataset movies ber-genre *drama* dan *comedy*, hal tersebut menunjukkan bahwa film yang tersedia lebih banyak ber-genre  _drama_  dan  _comedy_.  
   
 - **Rating**
 ![unv-3](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/img/EDA-Rating.png?raw=true)    
 Gambar 4. Visualisasi fitur numerik rating  
   Dari hasil visualisasi pada gambar 4 dapat disimpulkan bahwa:  
   - Rentang rating film adalah 0,5 hingga 5
   - Jumlah sampel terbanyak adalah film yang memiliki rating 4, hal ini menunjukkan bahwa banyak user yang menilai film dengan nilai 4.

## Data Preparation

Berikut adalah langkah-langkah dalam melakukan persiapan data:

- Gabungkan Dataset dan Tangani Data yang Hilang
Tahap ini melibatkan penggabungan kedua dataset, movies.csv dan ratings.csv, menggunakan fungsi [*merge()*(https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html). Setelah digabungkan, data yang memiliki nilai kosong atau hilang akan dihapus menggunakan fungsi [*dropna()*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html) agar memudahkan proses berikutnya.

- Hapus Data Duplikat
Proses ini melibatkan penggunaan fungsi [*drop_duplicates()*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html) untuk menghapus entri data yang sama, mencegah duplikasi yang tidak diinginkan dan mengurangi kebingungan.

- Konversi Data Series Menjadi Format List
Tahap ini melibatkan penggunaan fungsi [*tolist()*](https://pandas.pydata.org/docs/reference/api/pandas.Series.tolist.html) untuk mengonversi data series ke dalam bentuk list, mempermudah proses pemodelan data di tahap selanjutnya.

## Modeling
Setelah persiapan data selesai, langkah selanjutnya adalah pembuatan model, yang melibatkan tahapan-tahapan berikut:
Melakukan Vektorisasi dengan TF-IDF
Pada tahap ini, data yang telah dipersiapkan dikonversi menjadi vektor menggunakan fungsi [tfidfvectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) dari library sklearn. Proses ini bertujuan untuk mengidentifikasi korelasi antara judul film dengan kategori genre film.

 **Kelebihan**:
 - Mengatasi Masalah Dimensi: TF-IDF dapat mengurangi dimensi dataset dengan menghilangkan kata-kata umum yang tidak informatif.
 - Menangkap Bobot Kata: TF-IDF memberikan bobot yang lebih tinggi pada kata-kata yang jarang muncul tetapi penting, sehingga meningkatkan kemampuan model dalam membedakan antara dokumen.
 - Mudah diinterpretasikan: Hasil vektor TF-IDF relatif mudah diinterpretasikan, karena memberikan skor numerik yang menunjukkan tingkat pentingnya sebuah kata dalam dokumen.
 - Mengatasi Masalah Nilai Nol: TF-IDF menangani masalah nilai nol dengan baik, karena kata-kata yang tidak ada dalam dokumen akan memiliki nilai TF-IDF nol.

 **Kekurangan**:
 - Tidak Memperhatikan Urutan Kata: TF-IDF tidak memperhatikan urutan kata dalam dokumen, yang berarti informasi tentang struktur atau urutan kata dalam dokumen tidak dipertimbangkan.
 - Membutuhkan Banyak Data: Untuk hasil yang optimal, TF-IDF membutuhkan jumlah data yang cukup besar untuk memperoleh estimasi yang akurat tentang frekuensi kata-kata dalam dokumen.
 - Sensitif terhadap Stop Words: Penggunaan stop words yang berlebihan dapat mempengaruhi hasil vektorisasi TF-IDF, karena kata-kata tersebut dapat memiliki bobot yang signifikan tergantung pada frekuensinya dalam korpus.
 - Tidak Membedakan Makna: TF-IDF tidak membedakan makna kata yang sama tetapi digunakan dalam konteks yang berbeda. Misalnya, kata "batu" dapat merujuk pada benda fisik atau nama lokasi, tetapi TF-IDF mungkin memberikan bobot yang sama untuk kedua makna tersebut.

Mengukur Tingkat Kesamaan dengan Cosine Similarity [Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
Setelah data dikonversi menjadi vektor, langkah selanjutnya adalah mengukur tingkat kesamaan antara dua vektor menggunakan metode Cosine Similarity. Tujuan utamanya adalah untuk menentukan seberapa mirip dua vektor tersebut dengan melihat sudut kosinus antara keduanya. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity.

 **Kelebihan**:
 - Mudah diimplementasikan: Cosine Similarity adalah metode yang relatif mudah diimplementasikan dan dipahami. Ini sering digunakan dalam berbagai aplikasi pengolahan bahasa alami dan pemrosesan teks.
 - Tidak Sensitif Terhadap Magnitudo: Cosine Similarity tidak sensitif terhadap magnitudo vektor, yang berarti ia fokus pada arah dari vektor, bukan pada besarnya. Ini berguna dalam kasus di mana magnitudo data tidak relevan, seperti dalam pemrosesan teks.
 - Berfungsi Baik pada Data Sparse: Cosine Similarity efektif bekerja dengan data sparse, seperti data teks, di mana matriks term-frekuensi jarang (TF-IDF) sering digunakan untuk merepresentasikan dokumen.
 - Mengatasi Masalah Dimensi: Cosine Similarity dapat digunakan untuk mengatasi masalah dimensi dalam analisis data yang memiliki fitur yang sangat banyak.

 **Kekurangan**:
 - Tidak Memperhitungkan Konteks: Cosine Similarity hanya memperhitungkan arah vektor dan tidak mempertimbangkan konteks atau makna dari data yang diukur. Ini dapat menyebabkan hasil yang tidak sesuai dalam beberapa kasus.
 - Tidak Cocok untuk Data dengan Bobot Penting: Jika beberapa fitur memiliki bobot yang penting dalam menentukan kesamaan antara data, Cosine Similarity mungkin tidak memberikan penekanan yang cukup pada fitur-fitur tersebut.
 - Sensitif terhadap Bobot: Cosine Similarity bisa menjadi sensitif terhadap perbedaan dalam bobot yang diberikan pada fitur-fitur data. Ini dapat mempengaruhi hasil kesamaan, terutama jika bobot tidak diberikan secara tepat.
 - Tidak Membedakan Antar-Kategori: Cosine Similarity tidak secara otomatis dapat membedakan antara kategori data yang berbeda. Ini berarti bahwa dalam beberapa kasus, data dari kategori yang berbeda dapat memiliki kesamaan kosinus yang tinggi, meskipun sebenarnya berbeda dalam makna atau konteksnya.

Membuat Fungsi movie_recommendations()
Tahap terakhir dari proses pemodelan adalah pembuatan fungsi untuk menghasilkan rekomendasi top-N. Fungsi ini disebut *movie_recommendations()*. Fungsi ini menggunakan teknik [argpartition](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) untuk mengambil sejumlah nilai k tertinggi dari similarity data (dalam hal ini, dataframe cosine_sim_df). Data kemudian diurutkan dari nilai *similarity* tertinggi ke terendah, dan film yang sedang dicari [drop()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html) dihapus dari daftar rekomendasi agar tidak muncul kembali. Parameter fungsi *movie_recommendations()* adalah sebagai berikut:
  - movie_title : Judul film (index kemiripan dataframe)(str)
  - similarity_data : Dataframe kesamaan simetris dengan judul film sebagai indeks dan kolom (object)
  - items : Mengandung nama dan fitur lain yang digunakan untuk mendefinisikan kemiripan (object)
  - k : Jumlah rekomendasi yang ingin diberikan(int).

Setelah model selesai dibuat, panggil model untuk menampilkan hasil rekomendasi, sebagai contoh gunakan judul film *Baby Geniuses (1999)* untuk menguji model.  

|    |   id   |      movie_title     |   genre   |
|----|--------|----------------------|-----------|
|548 |  2555  | Baby Geniuses (1999) |  Comedy   | 

 Selanjutnya lihat rekomendasi film yang sesuai dengan genre yang sama dengan film *Baby Geniuses (1999)*.
 
| | movie_title  |   genre   |
|-|--------------|-----------|
|0| Andrew Dice Clay: Dice Rules (1991) | Comedy | 
|1| Hawks and Sparrows (Uccellacci e Uccellini) (1... | Comedy | 
|2| American Wedding (American Pie 3) (2003) | Comedy | 
|3| Uptown Girls (2003) | Comedy | 
|4| Start the Revolution Without Me (1970) | Comedy | 
|5| Legally Blonde 2: Red, White & Blonde (2003) | Comedy | 
|6| What's Up, Doc? (1972) | Comedy | 
|7| Kiss Me, Stupid (1964) | Comedy | 
|8| One, Two, Three (1961) | Comedy | 
|9| It's Pat (1994) | Comedy | 
    
 **Kelebihan**:
 - Kustomisasi Rekomendasi: Fungsi ini memungkinkan untuk membuat rekomendasi yang disesuaikan dengan preferensi pengguna atau kebutuhan bisnis tertentu, seperti menentukan jumlah rekomendasi yang diinginkan.
 - Mudah Digunakan: Dengan adanya fungsi ini, proses mendapatkan rekomendasi film menjadi lebih mudah dan efisien. Pengguna hanya perlu memasukkan judul film yang ingin dicari, dan fungsi akan menghasilkan rekomendasi yang sesuai.
 - Fleksibilitas: Fungsi ini dapat disesuaikan dengan berbagai jenis model rekomendasi dan metode pengukuran kesamaan, sehingga dapat digunakan dalam berbagai skenario dan kasus penggunaan.

 **Kekurangan**:
 - Ketergantungan pada Model: Kualitas rekomendasi dari fungsi ini sangat bergantung pada kualitas model rekomendasi yang digunakan. Jika model tidak akurat atau tidak sesuai dengan data, maka rekomendasi yang dihasilkan juga tidak akan akurat.
 - Kemungkinan Overfitting: Jika tidak hati-hati dalam pemilihan model atau parameter, ada risiko overfitting di mana model hanya cocok dengan data pelatihan tetapi tidak bisa melakukan generalisasi dengan baik pada data baru.
 - Keterbatasan dalam Pengukuran Kesamaan: Metode yang digunakan untuk mengukur kesamaan antara film-film mungkin tidak selalu sempurna atau mungkin tidak memperhitungkan semua faktor yang relevan dalam menentukan kesamaan.
 - Keterbatasan dalam Skala: Fungsi ini mungkin tidak optimal untuk digunakan pada skala besar dengan banyak pengguna dan film, karena dapat memakan waktu dan sumber daya yang signifikan untuk menghasilkan rekomendasi untuk setiap permintaan.

# Evaluation
---
Pada proyek ini, Metric yang digunakan pada sistem rekomendasi judul film berdasarkan genre adalah accuracy precision. Precision adalah metrik yang membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif dengan rumus

$$\ Precission=TP/(TP+FP)$$

Keterangan :

	TP = True Positif (prediksi positif dan hal tersebut benar)
 
	FP = False Positif (prediksi positif dan hal tersebut salah)

Berbeda dengan di modelling disini saya mengevaluasi dengan mencoba menampilkan 19 rekomendasi film dari judul yang telah di input yaitu Below (2002) genre horror, menggunakan fungsi movie_recomendations. Alasan accuracy Precision dipilih adalah karena metrik ini dapat membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif. Dalam hal ini adalah rasio item yang direkomendasikan memiliki genre yang mirip atau serupa dibandingkan dengan genre dari judul film yang diinput.

Fungsi yang digunakan untuk melihat jumlah genre yang mirip atau serupa yaitu dengan menggunakan value_counts().

Output:

|   |  genre  | count |  
|---|---------|-------|
| 0 |  Horror |  19   | 

Dipilih nya nilai True Positif 19 karna ia merupakan nilai atau jumlah yg diduga memiliki kemiripan/identik dengan genre yg dipilih yaitu 19. Hasil rekomendasi yg dihasilkan model menunjukan kemiripan dengan genre film yg dinput yaitu Horror. Sedangkan utk nilai False Positif tidak teridentifikasi pada hasil output dari genre yg diinput maka nilai nya 0. Setelah dihitung menggunakan rumus precision menghasilkan presisi sebesar 100%.
 
### Result  
Karena proyek ini menggunakan model Content-Based Filtering, metrik yang paling cocok untuk evaluasi adalah Precision. Berdasarkan hasil evalauasi, output yang dihasilkan bahwa prediksi rekomendasi yang diberikan 100% presisi sesuai genre yang mirip atau serupa dengan genre dari judul yang diinput. Oleh karena itu, nilai Precision dari model ini adalah 100%.

### Conclusion
Setelah melalui serangkaian proses yang komprehensif, mulai dari pengolahan dataset hingga evaluasi model, sistem rekomendasi menggunakan pendekatan Machine Learning Content-Based Filtering berhasil dibangun. Hasilnya memuaskan, di mana dari 10 judul film yang direkomendasikan, semua dianggap relevan dengan judul film yang diuji. Hal ini menunjukkan bahwa precision dari model ini mencapai 100%. Diharapkan implementasi sistem rekomendasi ini dapat membantu pengguna untuk menemukan berdasarkan genre film-film yang paling mirip dengan film yang telah ditonton sebelumnya dan dijadikan rekomendasi film yang akan ditonton selanjutnya. 

### REFERENCES
[1] F. Perdana, “SISTEM REKOMENDASI FILM MENGGUNAKAN ALGORITMA ITEM-BASED COLLABORATIVE FILTERING DAN BASIS DATA GRAPH,” Undergraduate thesis, 2017.[http://eprints.undip.ac.id/60611/](http://eprints.undip.ac.id/60611/).

[2] Setiani, Tia Dwi. "Machine Learning Terapan". Dicoding. 2021. [Online] Tersedia: [https://www.dicoding.com/academies/319/corridor](https://www.dicoding.com/academies/319/corridor) [Diakses pada 14 Maret 2024].

[3] Dicoding, 2024. Machine Learning Terapan | Sistem Rekomendasi. [Online]  Tersedia di: [https://www.dicoding.com/academies/319/tutorials/17109](https://www.dicoding.com/academies/319/tutorials/17109) [Diakses 16 Maret 2024]. 

‌[4] K. C. Media, “Pengertian Film: Definisi, Jenis dan Fungsinya Halaman all,” KOMPAS.com, Oct. 19, 2022.[https://entertainment.kompas.com/read/2022/10/19/150302666/pengertian-film-definisi-jenis-dan-fungsinya?page=all](https://entertainment.kompas.com/read/2022/10/19/150302666/pengertian-film-definisi-jenis-dan-fungsinya?page=all) [Diakses 17 Maret 2024].

