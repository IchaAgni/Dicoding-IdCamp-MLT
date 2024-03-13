# Laporan Proyek Machine Learning - Nisa Agni Afifah

## Domain Proyek

Topik proyek ini yaitu pada domain ekonomi dan bisnis, dengan berfokus pada rekrutmen karyawan baru oleh suatu perusahaan. Perusahaan ini tertarik untuk menilai kisaran gaji yang cocok berdasarkan pengalaman kerja calon pelamar yang akan direkrut.

### Latar Belakang
Sumber Daya Manusia (SDM) salah satu komponen yang berperan penting bagi
perusahaan, memegang peranan paling krusial dalam menentukan keberhasilan suatu tugas di dalam organisasi atau perusahaan. Dalam
mewujudkan tujuan yang optimal tentunya perusahaan
memerlukan tenaga suatu karyawan [[Pengaruh Gaji Dan Insentif Terhadap Kinerja Karyawan]](https://jurnal.stiekma.ac.id/index.php/JAMIN/article/view/53).

Dengan begitu perlu bagi perusahaan untuk memperhatikan para pekerjanya. Salah satu usaha untuk meningkatkan mutu SDM yaitu dengan pemberian gaji berdasarkan pengalaman kerja. Ketika seseorang yang sudah lama bekerja di suatu perusahaan maka gajinya akan semakin naik. Proyek ini ditujukan guna
menganalisis prediksi gaji karyawan berdasarkan lama tahun bekerja [[Prediksi Gaji Berdasarkan Pengalaman Bekerja Menggunakan Metode Regresi Linear]](https://doi.org/10.20895/dinda.v2i2.548).

Dalam artikel berjudul “Competing on Analytics” yang diterbitkan oleh [[Harvard Business Review]](https://hbr.org/2006/01/competing-on-analytics), Davenport, seorang ahli *business analytics* berpendapat bahwa senjata strategis di bidang bisnis saat ini adalah pengambilan keputusan analitik. Ia merupakan teknik pengambilan keputusan berdasarkan berbagai informasi yang diekstrak dari data.

> Masalah ini diselasaikan dengan menghasilkan model yang dapat memprediksi kisaran gaji calon karyawan di perusahaan berdasarkan tahun pengalaman bekerja
Perusahaan akan mencoba menerapkan 2 model Machine Learning dan kemudian memilih model yang prediksinya paling mendekati. 

Referensi yang di gunakan :
- [Dicoding, Machine Learning Terapan ](https://www.dicoding.com/academies/319/corridor)
- [Predictive analytics](https://www.ibm.com/analytics/predictive-analytics)


## Business Understanding
Rekrutmen adalah proses mencari dan menyeleksi calon karyawan untuk mengisi posisi atau jabatan tertentu. Kunci utama menciptakan Sumber Daya Manusia (SDM) yang profesional terletak pada proses rekrutmen, seleksi, training, dan development calon karyawan. Tidaklah mudah mencari karyawan yang berkualitas. Proses rekrutmen ini penting dalam menentukan baik tidaknya pelamar yang melamar pekerjaan pada suatu perusahaan [[Rekrutmen]](https://majoo.id/solusi/detail/rekrutmen-adalah#:~:text=Rekrutmen%20adalah%20proses%20mencari%20dan,mudah%20mencari%20karyawan%20yang%20berkualitas).

Faktor yang dapat menjadi penentu efektivitas seorang karyawan adalah pengalaman kerja; semakin besar pengalamannya, semakin tinggi pula kualitasnya. Dikutip dari salah satu Jurnal Ekonomi Pembangunan, Manajemen dan Bisnis, Akuntansi “Pengalaman kerja karyawan berhubungan erat dengan kinerja”.Pengalaman kerja akan membantu kelancaran didalam menyelesaikan pekerjaan dalam suatu perusahaan [[Dampak Masa Kerja, Pengalaman Kerja, Kemampuan Kerja Terhadap Kinerja Karyawan]](https://e-journal.upr.ac.id/index.php/jemba/article/download/2986/2501).

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
Solusi yang dapat dilakukan untuk memenuhi goals proyek ini diantaranya sebagai berikut:
- Membuat 2 model Machine Learning yaitu dengan algoritma LinearRegression dan RandomForest.
   * Konsep dasar dari algoritma Linear Regression adalah melakukan prediksi nilai y berdasarkan nilai x yang diketahui, sambil mencari nilai m dan b sehingga kesalahan (error) dapat diminimalkan sebanyak mungkin [[Simple Linear Regression di Python]](https://medium.com/@adiptamartulandi/belajar-machine-learning-simple-linear-regression-di-python-e82972695eaf)

     ![image](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/rumus.png)
     
Metode ini memiliki kelebihan ketika digunakan untuk memprediksi nilai yang ada di masa depan, terutama jika hubungan antara variabel independen dan dependen bersifat linear. Namun, kekurangan metode ini terletak pada keterbatasannya ketika variabel dependen dan independen jarang menunjukkan hubungan yang jelas dalam situasi sebenarnya [[LinearRegression]](https://caraguna.com/apa-itu-linear-regression-dalam-machine-learning/)

 ![image](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/LinearRegression.png)
   * Konsep dari algoritma RandomForest yaitu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Kelebihan dari metode ini yakni jika dataset berjumlah banyak maka RandomForest akan bekerja secara efisien.

![image](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/tree.png)
Tujuan yang ingin dicapai adalah estimasi gaji, dan sebagai yang kita ketahui, estimasi gaji merupakan variabel yang bersifat kontinu. Dalam konteks prediksi variabel kontinu, ini dapat dianggap sebagai masalah regresi. Dalam hal regresi seperti ini, metrik Mean Squared Error (MSE) akan digunakan. Secara keseluruhan, MSE mengukur sejauh mana prediksi mendekati nilai aktual. Oleh karena itu, setiap model akan dievaluasi, dan kemudian algoritma yang memberikan nilai metrik terbaik akan dipilih.
 
## Data Understanding
Dataset yang digunakan pada proyek ini dibuat oleh [[RubyDoby]](https://www.kaggle.com/rubydoby) yang di upload ke [Kaggle](https://www.kaggle.com/) Sumber dataset: [[Years of experience and employees salary]](https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary).

Pada berkas yang diunduh pada [link tersebut](https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary) yaitu [employee_salaries.csv](https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary?select=employee_salaries.csv) terdapat 1500 baris dan 2 kolom. 

### Variabel-variabel pada Years of experience and employee salary dataset adalah sebagai berikut:
- Years of Experience: total tahun pengalaman kerja.
- Salary: total gaji karyawan per tahun dalam kurs dollar.

### Exploratory Data Analysis - Univariate Analysis
![univariate](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/uni-years%20of%20experience.png)
Dari hasil visualisasi di atas dapat disimpulkan bahwa: 
- Sebagian besar sampel Years of experience berada di kisaran 8-14 tahun.
- Sebagian besar sampel Salary berada di kisaran 86000-90000.

### Exploratory Data Analysis - Multivariate Analysis
![multivariate-1](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/multivariate.png)   
Dari hasil visualisasi data di atas dapat disimpulkan bahwa:
-  Pola sebaran data pada grafik pairplot di atas memiliki korelasi posistif

![multivariate-2](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/heatmap.png)  
Berdasarkan visualisasi heatmap di atas dapat disimpulkan bahwa:
- Variabel Years of experience berkorelasi positif dengan variabel Salary, skornya yaitu 0.8.

## Data Preparation
Berikut merupakan tahapan-tahapan dalam Data Preparation:
- Data yang ada akan dipisah menjadi dua bagian, yaitu data latih dan data uji, dengan proporsi 80:10. Proses ini dilakukan menggunakan modul [[train_test_split]] (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) dari library scikit-learn.
- Data latih akan di standarisasi menggunakan StandardScaler dari library scikit-learn.
  
### Alasan memilih tahapan tersebut di data preparation yaitu : 
Train Test Split :
- Evaluasi Kinerja Model
- Mencegah Overfitting
- Validasi Model
  
Standarisasi :
- Mengatasi masalah skala
- Percepatan Konvergensi Algoritma
- Menghindari Bias
- Interpretabilitas Model
  
Penerapan train-test split dan standarisasi dapat membantu memastikan bahwa model yang dihasilkan dapat diandalkan, umum, dan tidak dipengaruhi oleh masalah yang mungkin muncul akibat pembagian data yang tidak benar atau perbedaan skala variabel yang signifikan.

## Modeling
Setelah menyelesaikan tahapan data preparation, langkah selanjutnya adalah membuat dua model sebagai perbandingan. Pertama, akan dibuat model menggunakan algoritma Linear Regression [[LinearRegression]](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). Algoritma ini dipilih karena umum digunakan untuk menyelesaikan permasalahan regresi, serta memiliki kelebihan dalam kemudahan pemahaman. Model Linear Regression efektif untuk memodelkan hubungan linier antara variabel input dan output.

Kedua, model akan dibuat menggunakan algoritma RandomForest [[RandomForest]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Kelebihan dari algoritma ini termasuk kemampuannya untuk mengatasi noise dan nilai yang hilang, serta efektif mengelola dataset dalam skala besar. Meskipun demikian, Random Forest juga memiliki kekurangan, seperti interpretasi yang sulit dan kebutuhan tuning model yang cermat untuk mendapatkan hasil yang optimal, terutama pada dataset yang kompleks.

## Evaluation
Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut. Dalam proyek ini, evaluasi model dilakukan menggunakan metrik Mean Squared Error (MSE). Metrik ini mengukur jumlah rata-rata selisih kuadrat antara nilai aktual dan nilai prediksi [[Evaluasi Model]](https://www.dicoding.com/academies/319/tutorials/18595).

MSE didefinisikan dalam persamaan berikut:

![image](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/mse.jpeg)  
Keterangan:  
N = jumlah dataset  
yi = nilai sebenarnya  
y_pred = nilai prediksi

### Model dengan Algoritma LinearRegression  
![LR-1](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/LR-1.png)  
Seperti terlihat pada gambar, model yang dibuat menggunakan algortima LinearRegression memiliki nilai MSE yang sangat tinggi hingga mencapai 113458.095232	pada saat training dan 127004.94334 pada saat test, hal ini menunjukkan algoritma ini kurang baik untuk melakukan prediksi.  

![image](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/LR2.jpg)  

Pada proses pengujian pun dapat terlihat hasil prediksi tidak akurat dengan nilai sebenarnya.

### Model dengan Algortima RandomForest  
![RF-1](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/RF1.png)  
Seperti terlihat pada gambar, saat dibandingkan dengan algoritma LinearRegression terlihat algortima RandomForest memiliki nilai MSE yang lebih rendah yaitu 13235.129443 pada saat training dan	15922.675464 pada saat test, hal ini menunjukkan algoritma RandomForest lebih baik untuk melakukan prediksi dibanding LinearRegression.  

![image](https://github.com/IchaAgni/Dicoding-IdCamp-MLT/blob/main/image/RF2.jpg)  

Pada proses pengujian dapat terlihat hasil prediksi dari model yang menggunakan RandomForest lebih akurat dengan nilai sebenarnya. Oleh karena itu algoritma ini yang akan dipilih sebagai model utama untuk memprediksi kisaran gaji karyawan.

