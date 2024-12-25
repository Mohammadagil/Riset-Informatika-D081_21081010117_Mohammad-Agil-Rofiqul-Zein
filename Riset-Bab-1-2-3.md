# Analisis Performansi Model VGG-16 dengan Metode Klasifikasi ELM Berdasarkan Variasi Input Citra dan Komposisi Warna HSV untuk Klasifikasi Pneumonia

BAB I
PENDAHULUAN

1.1 Latar Belakang
Paru-paru adalah organ pernapasan yang berperan dalam sistem pernapasan dan sirkulasi darah pada makhluk hidup yang bernapas menggunakan udara. Fungsi utamanya adalah mengeluarkan karbon dioksida dari aliran darah ke udara. Proses pernapasan dimulai saat udara masuk melalui hidung atau mulut, lalu melewati trakea (tenggorokan), bronkus, bronkiolus, hingga mencapai alveolus. Alveolus berfungsi menyebarkan oksigen ke seluruh tubuh, sementara karbon dioksida dikeluarkan dari tubuh. 
Namun, sistem pernapasan sangat rentan terhadap berbagai gangguan yang dapat disebabkan oleh infeksi, polusi udara, gaya hidup yang buruk, dan faktor lingkungan lainnya. Penyakit seperti asma, bronkitis, dan pneumonia menjadi beberapa kondisi yang paling umum memengaruhi fungsi pernapasan. Menurut Organisasi Kesehatan Dunia (WHO), polusi udara merupakan salah satu penyebab utama gangguan pernapasan, dengan sekitar 91% populasi global tinggal di daerah yang kualitas udaranya tidak memenuhi standar kesehatan yang aman. Hal ini menunjukkan bahwa faktor eksternal seperti kualitas udara memiliki dampak yang signifikan terhadap kesehatan pernapasan 
Di era modern ini, Paparan polusi udara, seperti partikel halus (PM2.5), nitrogen dioksida (NO₂), dan ozon (O₃), dapat memicu berbagai masalah kesehatan, termasuk gangguan pada fungsi paru-paru, risiko yang lebih tinggi terkena penyakit pernapasan kronis, serta gangguan kardiovaskular. Selain itu, polusi udara juga menjadi faktor yang memperburuk kondisi kesehatan, yang dapat meningkatkan angka kematian, terutama di daerah perkotaan dengan tingkat industrialisasi yang tinggi. 
Pemahaman yang lebih mendalam tentang mekanisme sistem pernapasan dan berbagai faktor yang memengaruhinya menjadi sangat penting. Selain itu, inovasi teknologi, seperti penerapan kecerdasan buatan dalam analisis citra medis paru-paru, diharapkan dapat mempercepat diagnosis dini dan pengelolaan penyakit pernapasan. Langkah ini bertujuan untuk meningkatkan kualitas hidup pasien serta menurunkan angka kematian akibat gangguan pernapasan, sekaligus menjadi bagian penting dari upaya global dalam meningkatkan kesehatan masyarakat. 
Untuk meningkatkan efisiensi dan akurasi diagnosis pneumonia, diperlukan pendekatan otomatis berbasis teknologi yang andal. Salah satu teknologi yang berkembang pesat dalam analisis citra medis adalah Convolutional Neural Network (CNN). Sebagai bagian dari pembelajaran mendalam (deep learning), CNN dirancang untuk secara otomatis mengekstraksi fitur kompleks dari citra melalui lapisan-lapisan konvolusi. Beberapa arsitektur CNN telah banyak digunakan untuk klasifikasi citra medis, termasuk VGG-16, AlexNet dan GoogleNet, yang dikenal dengan keunggulannya dalam tugas-tugas klasifikasi.
VGG-16 adalah arsitektur Convolutional Neural Network (CNN) yang dikembangkan oleh Visual Geometry Group dari Universitas Oxford. Arsitektur ini terdiri dari 16 lapisan yang dapat dilatih, termasuk 13 lapisan konvolusi dan 3 lapisan fully connected. VGG-16 dikenal karena penggunaan filter konvolusi berukuran kecil (3×3) secara konsisten di seluruh jaringan, yang memungkinkan pendalaman jaringan tanpa meningkatkan jumlah parameter secara signifikan. 
VGG-16 telah diterapkan secara luas dalam deteksi pneumonia melalui analisis citra X-ray dada. Arsitektur ini, dengan 16 lapisan yang dapat dilatih, efektif dalam mengekstraksi fitur kompleks dari citra medis. Beberapa penelitian dalam lima tahun terakhir menunjukkan keberhasilan penggunaan VGG-16 dalam tugas ini. Misalnya, sebuah studi pada tahun 2021 mengembangkan model VGG-16 yang disesuaikan untuk mendeteksi pneumonia menggunakan pendekatan transfer learning. Model ini mencapai akurasi tinggi dalam klasifikasi citra X-ray dada antara pneumonia dan kondisi normal. 
Penggabungan arsitektur CNN yang lebih kompleks, seperti VGG-16, dengan metode pembelajaran cepat seperti Extreme Learning Machine (ELM) telah menjadi solusi yang diusulkan untuk meningkatkan efisiensi pelatihan dan keakuratan model. VGG-16, sebagai model CNN dengan 16 lapisan, telah terbukti berhasil dalam berbagai tugas klasifikasi citra.
ELM, dengan karakteristiknya yang ringan dan efisien, dapat menggantikan lapisan klasifikasi tradisional pada CNN, sehingga mempercepat proses pelatihan tanpa mengurangi tingkat akurasi. Selain itu, analisis berbagai ukuran dan variasi komposisi warna pada citra input dapat membantu mengidentifikasi kombinasi terbaik untuk meningkatkan efisiensi penggunaan metode CNN. Pendekatan ini membuat model lebih responsif terhadap variasi data input, yang pada akhirnya dapat memperbaiki performa klasifikasi. Secara keseluruhan, kombinasi dari arsitektur CNN yang dalam, metode pembelajaran cepat seperti ELM, dan optimasi parameter input citra memiliki potensi besar untuk meningkatkan efisiensi dan akurasi dalam model klasifikasi citra. 
Untuk mendukung penelitian ini, peneliti membutuhkan data citra X-ray dada yang andal dan berkualitas untuk digunakan dalam proses pelatihan dan pengujian model klasifikasi pneumonia. Salah satu sumber data yang kredibel adalah platform Kaggle, yang menyediakan dataset publik berupa citra X-ray dada yang telah dikurasi dan digunakan secara luas dalam berbagai penelitian terkait analisis citra medis. Dataset ini tidak hanya mencakup citra dari pasien dengan pneumonia, tetapi juga dari individu dengan kondisi paru-paru normal, sehingga memungkinkan proses pelatihan model menjadi lebih terarah dan akurat. Dengan memanfaatkan dataset dari Kaggle, penelitian ini diharapkan dapat memperoleh data dengan variasi yang cukup untuk meningkatkan generalisasi model serta mendukung evaluasi performa sistem klasifikasi yang dikembangkan.

1.2 Rumusan Masalah
1. Bagaimana hasil performansi model VGG-16 dengan metode klasifikasi ELM berdasarkan variasi input citra dan komposisi warna HSV untuk klasifikasi pneumonia.
2. Bagaimana pengaruh variasi input citra dan komposisi warna HSV untuk tingkat performansi klasifikasi pneumonia.

1.3 Tujuan Penelitian
1. Membuat model pembelajaran CNN dengan arsitektur VGG-16 dan Ectreme Learning Machine
    (ELM), berdasarkan variasi input citra dan komposisi warna HSV untuk klasifikasi pneumonia.
2. Menganalisa hasil pengaruh variasi input citra dan komposisi warna HSV untuk tingkat performansi klasifikasi pneumonia.

1.4 Manfaat Penelitian
Penelitian ini diharapkan dapat memberikan manfaat sebagai berikut:
1.	Manfaat Akademik
•	Menambah literatur dan wawasan ilmiah tentang penerapan metode CNN (VGG-16) yang dikombinasikan dengan Extreme Learning Machine (ELM) dalam klasifikasi citra medis, khususnya pneumonia.
•	Menjadi referensi bagi penelitian selanjutnya yang berkaitan dengan pengembangan diagnosis berbasis kecerdasan buatan dalam bidang kesehatan.
2.	Manfaat Praktis
•	Memberikan solusi diagnosis otomatis yang efisien dan akurat untuk mendeteksi pneumonia melalui citra X-ray dada, yang dapat membantu tenaga medis dalam pengambilan keputusan klinis.
•	Mengurangi waktu dan biaya yang diperlukan untuk proses diagnosis manual, sehingga meningkatkan efisiensi pelayanan kesehatan.

3.	Manfaat Teknologi
•	Mengembangkan aplikasi kecerdasan buatan yang mengintegrasikan kekuatan CNN dalam ekstraksi fitur citra dengan efisiensi ELM, yang dapat diaplikasikan lebih luas dalam sistem pelayanan kesehatan modern.
•	Mendorong pemanfaatan teknologi deep learning dan machine learning dalam menghadapi tantangan kesehatan masyarakat, khususnya di daerah dengan keterbatasan sumber daya medis.

1.5 Batasan Masalah
Untuk memastikan fokus penelitian, batasan masalah ditentukan sebagai berikut:
1.	Penelitian ini hanya menggunakan model CNN dengan arsitektur VGG-16, serta mengintegrasikan Extreme Learning Machine (ELM) untuk lapisan klasifikasinya. Model atau metode lain di luar ini tidak dibahas.
2.	Data yang digunakan adalah citra X-ray dada dari dataset publik yang tersedia di platform Kaggle, dengan asumsi bahwa data tersebut sudah divalidasi dan dikurasi untuk keperluan penelitian.
3.	Analisis performa model dibatasi pada metrik pengujian tingkat akurasi, presisi, recall, f1-score, dan confusion matrix.
4.	Penelitian ini berfokus pada klasifikasi dua kategori utama, yaitu citra dengan kondisi normal dan citra dengan pneumonia. Klasifikasi penyakit paru-paru lainnya tidak termasuk dalam ruang lingkup penelitian.
Batasan ini ditetapkan untuk menjaga lingkup penelitian tetap terarah, terukur, dan dapat diselesaikan dalam waktu yang telah direncanakan.




