# Analisis Performansi Model VGG-16 dengan Metode Klasifikasi ELM Berdasarkan Variasi Input Citra dan Komposisi Warna HSV untuk Klasifikasi Pneumonia

## BAB I
## PENDAHULUAN

#### 1.1 Latar Belakang
Paru-paru adalah organ pernapasan yang berperan dalam sistem pernapasan dan sirkulasi darah pada makhluk hidup yang bernapas menggunakan udara. Fungsi utamanya adalah mengeluarkan karbon dioksida dari aliran darah ke udara. Proses pernapasan dimulai saat udara masuk melalui hidung atau mulut, lalu melewati trakea (tenggorokan), bronkus, bronkiolus, hingga mencapai alveolus. Alveolus berfungsi menyebarkan oksigen ke seluruh tubuh, sementara karbon dioksida dikeluarkan dari tubuh. 

  Namun, sistem pernapasan sangat rentan terhadap berbagai gangguan yang dapat disebabkan oleh infeksi, polusi udara, gaya hidup yang buruk, dan faktor lingkungan lainnya. Penyakit seperti asma, bronkitis, dan pneumonia menjadi beberapa kondisi yang paling umum memengaruhi fungsi pernapasan. Menurut Organisasi Kesehatan Dunia (WHO), polusi udara merupakan salah satu penyebab utama gangguan pernapasan, dengan sekitar 91% populasi global tinggal di daerah yang kualitas udaranya tidak memenuhi standar kesehatan yang aman. Hal ini menunjukkan bahwa faktor eksternal seperti kualitas udara memiliki dampak yang signifikan terhadap kesehatan pernapasan.
  
Di era modern ini, Paparan polusi udara, seperti partikel halus (PM2.5), nitrogen dioksida (NO₂), dan ozon (O₃), dapat memicu berbagai masalah kesehatan, termasuk gangguan pada fungsi paru-paru, risiko yang lebih tinggi terkena penyakit pernapasan kronis, serta gangguan kardiovaskular. Selain itu, polusi udara juga menjadi faktor yang memperburuk kondisi kesehatan, yang dapat meningkatkan angka kematian, terutama di daerah perkotaan dengan tingkat industrialisasi yang tinggi.

Pemahaman yang lebih mendalam tentang mekanisme sistem pernapasan dan berbagai faktor yang memengaruhinya menjadi sangat penting. Selain itu, inovasi teknologi, seperti penerapan kecerdasan buatan dalam analisis citra medis paru-paru, diharapkan dapat mempercepat diagnosis dini dan pengelolaan penyakit pernapasan. Langkah ini bertujuan untuk meningkatkan kualitas hidup pasien serta menurunkan angka kematian akibat gangguan pernapasan, sekaligus menjadi bagian penting dari upaya global dalam meningkatkan kesehatan masyarakat. 

Untuk meningkatkan efisiensi dan akurasi diagnosis pneumonia, diperlukan pendekatan otomatis berbasis teknologi yang andal. Salah satu teknologi yang berkembang pesat dalam analisis citra medis adalah Convolutional Neural Network (CNN). Sebagai bagian dari pembelajaran mendalam (deep learning), CNN dirancang untuk secara otomatis mengekstraksi fitur kompleks dari citra melalui lapisan-lapisan konvolusi. Beberapa arsitektur CNN telah banyak digunakan untuk klasifikasi citra medis, termasuk VGG-16, AlexNet dan GoogleNet, yang dikenal dengan keunggulannya dalam tugas-tugas klasifikasi.

VGG-16 adalah arsitektur Convolutional Neural Network (CNN) yang dikembangkan oleh Visual Geometry Group dari Universitas Oxford. Arsitektur ini terdiri dari 16 lapisan yang dapat dilatih, termasuk 13 lapisan konvolusi dan 3 lapisan fully connected. VGG-16 dikenal karena penggunaan filter konvolusi berukuran kecil (3×3) secara konsisten di seluruh jaringan, yang memungkinkan pendalaman jaringan tanpa meningkatkan jumlah parameter secara signifikan. 

VGG-16 telah diterapkan secara luas dalam deteksi pneumonia melalui analisis citra X-ray dada. Arsitektur ini, dengan 16 lapisan yang dapat dilatih, efektif dalam mengekstraksi fitur kompleks dari citra medis. Beberapa penelitian dalam lima tahun terakhir menunjukkan keberhasilan penggunaan VGG-16 dalam tugas ini. Misalnya, sebuah studi pada tahun 2021 mengembangkan model VGG-16 yang disesuaikan untuk mendeteksi pneumonia menggunakan pendekatan transfer learning. Model ini mencapai akurasi tinggi dalam klasifikasi citra X-ray dada antara pneumonia dan kondisi normal. 

Penggabungan arsitektur CNN yang lebih kompleks, seperti VGG-16, dengan metode pembelajaran cepat seperti Extreme Learning Machine (ELM) telah menjadi solusi yang diusulkan untuk meningkatkan efisiensi pelatihan dan keakuratan model. VGG-16, sebagai model CNN dengan 16 lapisan, telah terbukti berhasil dalam berbagai tugas klasifikasi citra.

ELM, dengan karakteristiknya yang ringan dan efisien, dapat menggantikan lapisan klasifikasi tradisional pada CNN, sehingga mempercepat proses pelatihan tanpa mengurangi tingkat akurasi. Selain itu, analisis berbagai ukuran dan variasi komposisi warna pada citra input dapat membantu mengidentifikasi kombinasi terbaik untuk meningkatkan efisiensi penggunaan metode CNN. Pendekatan ini membuat model lebih responsif terhadap variasi data input, yang pada akhirnya dapat memperbaiki performa klasifikasi. Secara keseluruhan, kombinasi dari arsitektur CNN yang dalam, metode pembelajaran cepat seperti ELM, dan optimasi parameter input citra memiliki potensi besar untuk meningkatkan efisiensi dan akurasi dalam model klasifikasi citra. 

Untuk mendukung penelitian ini, peneliti membutuhkan data citra X-ray dada yang andal dan berkualitas untuk digunakan dalam proses pelatihan dan pengujian model klasifikasi pneumonia. Salah satu sumber data yang kredibel adalah platform Kaggle, yang menyediakan dataset publik berupa citra X-ray dada yang telah dikurasi dan digunakan secara luas dalam berbagai penelitian terkait analisis citra medis. Dataset ini tidak hanya mencakup citra dari pasien dengan pneumonia, tetapi juga dari individu dengan kondisi paru-paru normal, sehingga memungkinkan proses pelatihan model menjadi lebih terarah dan akurat. Dengan memanfaatkan dataset dari Kaggle, penelitian ini diharapkan dapat memperoleh data dengan variasi yang cukup untuk meningkatkan generalisasi model serta mendukung evaluasi performa sistem klasifikasi yang dikembangkan.

#### 1.2 Rumusan Masalah
1. Bagaimana hasil performansi model VGG-16 dengan metode klasifikasi ELM berdasarkan variasi input citra dan komposisi warna HSV untuk klasifikasi pneumonia.
2. Bagaimana pengaruh variasi input citra dan komposisi warna HSV untuk tingkat performansi klasifikasi pneumonia.

#### 1.3 Tujuan Penelitian
1. Membuat model pembelajaran CNN dengan arsitektur VGG-16 dan Ectreme Learning Machine
    (ELM), berdasarkan variasi input citra dan komposisi warna HSV untuk klasifikasi pneumonia.
2. Menganalisa hasil pengaruh variasi input citra dan komposisi warna HSV untuk tingkat performansi klasifikasi pneumonia.

#### 1.4 Manfaat Penelitian
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

#### 1.5 Batasan Masalah
Untuk memastikan fokus penelitian, batasan masalah ditentukan sebagai berikut:
1.	Penelitian ini hanya menggunakan model CNN dengan arsitektur VGG-16, serta mengintegrasikan Extreme Learning Machine (ELM) untuk lapisan klasifikasinya. Model atau metode lain di luar ini tidak dibahas.
2.	Data yang digunakan adalah citra X-ray dada dari dataset publik yang tersedia di platform Kaggle, dengan asumsi bahwa data tersebut sudah divalidasi dan dikurasi untuk keperluan penelitian.
3.	Analisis performa model dibatasi pada metrik pengujian tingkat akurasi, presisi, recall, f1-score, dan confusion matrix.
4.	Penelitian ini berfokus pada klasifikasi dua kategori utama, yaitu citra dengan kondisi normal dan citra dengan pneumonia. Klasifikasi penyakit paru-paru lainnya tidak termasuk dalam ruang lingkup penelitian.
Batasan ini ditetapkan untuk menjaga lingkup penelitian tetap terarah, terukur, dan dapat diselesaikan dalam waktu yang telah direncanakan.

## BAB II
## TINJAUAN PUSTAKA

### 2.1 	Penelitian Sebelumnya
Dalam menjalankan sebuah penelitian, penting untuk merujuk pada penelitian-penelitian sebelumnya yang relevan agar dapat memahami landasan teori dan metode yang telah diterapkan. Penelitian mengenai klasifikasi pneumonia menggunakan deep learning, khususnya Convolutional Neural Networks (CNN), telah dilakukan oleh banyak peneliti dengan pendekatan dan dataset yang berbeda. Dengan meninjau penelitian-penelitian sebelumnya, dapat diperoleh wawasan tentang perbedaan dan kekurangan dalam metode yang ada, yang kemudian dapat dijadikan acuan dalam penelitian baru. 

Salah satu contoh penelitian yang relevan adalah penelitian oleh Rahman et al. pada tahun 2021 yang berjudul “Klasifikasi Pneumonia Menggunakan Deep Learning dengan Arsitektur CNN”. Penelitian ini mengkaji penggunaan arsitektur CNN untuk mendeteksi pneumonia dengan dataset X-ray dada yang terdiri dari 5.856 gambar. Model CNN yang digunakan adalah VGG-16, dengan variasi ukuran input citra untuk mengevaluasi pengaruh ukuran citra terhadap akurasi klasifikasi. Hasil penelitian ini menunjukkan bahwa ukuran citra 224x224 piksel memberikan akurasi klasifikasi sebesar 96,85% (Rahman et al., 2021).

Penelitian lain dilakukan oleh Zhang et al. pada tahun 2020 dengan judul “Penggunaan CNN untuk Klasifikasi Pneumonia Berdasarkan Citra X-ray Dada dengan Optimasi Lapisan Konvolusi”. Penelitian ini bertujuan untuk meningkatkan akurasi deteksi pneumonia dengan melakukan optimasi pada lapisan konvolusi dan pooling dalam arsitektur CNN. Dataset yang digunakan terdiri dari 6.000 gambar X-ray dada, dan penelitian ini berhasil mencapai akurasi klasifikasi sebesar 97,2%. Hasil tersebut menunjukkan bahwa dengan modifikasi arsitektur CNN, akurasi klasifikasi pneumonia dapat ditingkatkan lebih lanjut (Zhang et al., 2020).

Selain itu, penelitian oleh Li et al. pada tahun 2022 berjudul “Peningkatan Akurasi Klasifikasi Pneumonia Menggunakan Arsitektur VGG-16 dan Transfer Learning” mengkaji penerapan metode transfer learning pada VGG-16 untuk klasifikasi pneumonia. Dengan menggunakan dataset dari Kaggle yang terdiri dari 5.000 gambar X-ray dada, penelitian ini menemukan bahwa transfer learning dapat meningkatkan akurasi klasifikasi, bahkan dengan jumlah data yang terbatas. Model yang dihasilkan mencapai akurasi sebesar 98,4%. Pendekatan ini membuktikan bahwa pemanfaatan model yang telah dilatih sebelumnya dapat meningkatkan performa pada tugas-tugas medis (Li et al., 2022).

Selanjutnya, penelitian oleh Ahmed et al. (2021) dengan judul “Analisis Perbandingan Arsitektur CNN untuk Klasifikasi Pneumonia dengan Variasi Ukuran Input Citra” mengkaji pengaruh variasi ukuran input citra terhadap hasil klasifikasi pneumonia. Penelitian ini menggunakan beberapa arsitektur CNN seperti AlexNet, VGG-16, dan ResNet-50 dengan ukuran citra yang berbeda, mulai dari 128x128 hingga 512x512 piksel. Hasil dari penelitian ini menunjukkan bahwa model VGG-16 dengan ukuran citra 224x224 piksel menghasilkan akurasi tertinggi sebesar 97,5%, mengindikasikan bahwa ukuran input yang optimal sangat penting dalam meningkatkan kinerja model CNN dalam deteksi pneumonia (Ahmed et al., 2021).

Terakhir, penelitian oleh Kumar et al. (2023) berjudul “Pengaruh Komposisi Warna HSV terhadap Akurasi Klasifikasi Pneumonia Menggunakan CNN” memfokuskan pada analisis pengaruh komposisi warna dalam citra X-ray dada untuk deteksi pneumonia. Dalam penelitian ini, digunakan ruang warna HSV untuk meningkatkan kontras dan detail citra, yang diharapkan dapat membantu model CNN dalam mengekstraksi fitur lebih baik. Dengan menggunakan dataset yang terdiri dari 4.000 gambar X-ray, penelitian ini berhasil meningkatkan akurasi model CNN sebesar 2-3% dibandingkan dengan menggunakan citra dalam format grayscale. Hasil penelitian ini menunjukkan bahwa komposisi warna dapat mempengaruhi hasil klasifikasi dalam aplikasi medis (Kumar et al., 2023).

Penelitian-penelitian di atas memberikan wawasan yang berharga dalam pengembangan model CNN untuk klasifikasi pneumonia, terutama mengenai penggunaan arsitektur yang mendalam seperti VGG-16, variasi ukuran citra, dan pemanfaatan komposisi warna untuk meningkatkan akurasi klasifikasi. Dalam penelitian ini, kami akan menganalisis performansi model VGG-16 dengan metode Extreme Learning Machine (ELM) berdasarkan variasi input citra dan komposisi warna HSV untuk klasifikasi pneumonia, dengan tujuan untuk memperoleh solusi yang lebih efisien dan akurat.

### 2.2 	Penyakit Pneumonia
Pneumonia adalah salah satu penyakit infeksi yang menyerang paru-paru dan dapat disebabkan oleh berbagai jenis mikroorganisme, termasuk bakteri, virus, jamur, atau parasit. Penyakit ini menyebabkan peradangan pada kantung udara kecil (alveoli) di paru-paru, yang berfungsi untuk pertukaran gas oksigen dan karbon dioksida. Pneumonia dapat mempengaruhi satu atau kedua paru-paru, dan tingkat keparahannya bervariasi mulai dari infeksi ringan hingga yang mengancam jiwa. Gejala umum pneumonia meliputi batuk, demam, sesak napas, rasa lelah, nyeri dada, dan penurunan nafsu makan. Pada kelompok rentan seperti bayi, lansia, atau individu dengan sistem kekebalan tubuh yang lemah, pneumonia dapat menyebabkan komplikasi serius yang dapat mengarah pada kegagalan organ atau kematian (Meyer et al., 2021).

Untuk mendeteksi pneumonia, citra medis, terutama menggunakan X-ray dada, adalah salah satu metode yang paling umum digunakan. Pada paru-paru yang sehat, tampak kontras yang jelas antara jaringan paru dan udara yang terisi di dalam alveoli. Pada citra X-ray, paru-paru sehat muncul gelap karena udara mengisi alveoli, sedangkan jaringan lunak atau organ lain tidak dapat terlihat dengan jelas. Namun, pada paru-paru yang terinfeksi pneumonia, perubahan struktur paru-paru terjadi, dan area infeksi atau peradangan akan terlihat lebih terang pada gambar X-ray karena akumulasi cairan, eksudat, dan sel-sel inflamasi yang mengisi alveoli. Infiltrasi atau konsolidasi ini mengurangi efisiensi pertukaran oksigen, menyebabkan gangguan pernapasan dan menurunnya kadar oksigen dalam darah (Li et al., 2023). 

![image](https://github.com/user-attachments/assets/1f4d9a1b-65a9-4852-ba7b-12b44f495b88) 

Gambar 2.1 Paru-paru Pneumonia

![image](https://github.com/user-attachments/assets/85f9c41a-4e2b-479c-8e4a-9491741474ba) 

Gambar 2.2 Paru-paru Normal

Pada penelitian oleh Zhang et al. (2020), para peneliti menjelaskan bahwa pneumonia, terutama yang disebabkan oleh infeksi bakteri, dapat menyebabkan infiltrasi berupa bayangan putih pada citra X-ray dada yang menunjukkan adanya cairan atau nanah di dalam alveoli. Hal ini berbeda dengan kondisi paru-paru yang sehat, di mana area paru-paru tampak jelas dengan pola gelap. Dengan adanya infiltrasi inflamasi, konsolidasi ini dapat terlihat sebagai pengaburan pada gambar X-ray, yang mempermudah identifikasi dan diagnosis pneumonia. Penelitian ini juga menyoroti pentingnya deteksi dini dengan pencitraan untuk memberikan penanganan yang cepat, terutama dalam kasus pneumonia yang lebih parah.

Paru-paru normal, yang sehat dan bebas dari infeksi, menunjukkan pola hitam atau gelap pada citra X-ray dada, yang mengindikasikan bahwa alveoli terisi udara. Hal ini memungkinkan pertukaran gas yang efisien. Dalam keadaan pneumonia, terutama pada pneumonia bakterial atau viral, perubahan struktural pada paru-paru dapat menyebabkan pengurangan area yang mengandung udara, sehingga citra X-ray menampilkan area yang lebih terang atau kabur. Cairan yang terkumpul dalam alveoli sebagai respons terhadap infeksi menghalangi penetrasi sinar-X, yang menghasilkan bayangan putih yang lebih terang pada gambar X-ray. Pada pneumonia berat, peradangan ini bisa melibatkan sebagian besar jaringan paru-paru, mengakibatkan pengaburan yang meluas dan mengurangi kemampuan paru-paru untuk melakukan pertukaran gas yang normal (Rahman et al., 2022).

Penelitian lain yang dilakukan oleh Li et al. (2022) mengungkapkan bahwa pengaburan atau konsolidasi pada X-ray dada merupakan ciri khas dari pneumonia yang disebabkan oleh infeksi bakteri atau virus. Para peneliti menekankan pentingnya evaluasi secara komprehensif terhadap pola bayangan yang ada di citra medis untuk membedakan pneumonia dari penyakit paru lainnya, seperti kanker paru atau penyakit paru obstruktif kronik (PPOK). Diagnosis yang tepat sangat penting untuk memastikan bahwa pasien menerima pengobatan yang sesuai. Oleh karena itu, analisis citra X-ray sangat krusial dalam mendiagnosis pneumonia, karena memberikan gambaran yang jelas tentang lokasi dan tingkat keparahan infeksi paru-paru (Kumar et al., 2023).

Selain itu, pneumonia yang disebabkan oleh infeksi bakteri sering kali mempengaruhi satu lobus paru, sementara pneumonia viral dapat menyebar lebih merata pada kedua paru-paru. Citra X-ray memberikan gambaran mengenai lokasi dan sejauh mana infeksi menyebar di paru-paru. Pada beberapa kasus, pneumonia dapat menyebabkan komplikasi lebih lanjut, seperti abses paru atau efusi pleura, yang juga dapat terdeteksi melalui citra medis. Penelitian oleh Kumar et al. (2023) menunjukkan bahwa deteksi awal dan penanganan yang cepat berdasarkan hasil pencitraan X-ray dapat mengurangi tingkat kematian akibat pneumonia, terutama pada kasus yang lebih parah.

Dengan demikian, pemahaman yang mendalam tentang perbedaan antara paru-paru yang sehat dan yang terinfeksi pneumonia sangat penting dalam proses diagnosis dan pengobatan. Deteksi dini melalui X-ray dada, di mana tanda-tanda peradangan atau konsolidasi paru terlihat jelas, dapat membantu dokter untuk melakukan diagnosis yang tepat dan memberikan terapi yang sesuai untuk pasien dengan pneumonia.

### 2.3 	Convolutional Neural Networks (CNN) dalam Klasifikasi Citra Medis
Convolutional Neural Networks (CNN) telah menjadi model pembelajaran mendalam yang sangat populer dalam pengolahan citra medis, khususnya untuk tugas klasifikasi penyakit berdasarkan citra medis. CNN sangat efektif dalam mengekstraksi fitur secara otomatis dari citra, yang memudahkan identifikasi pola-pola penting dalam gambar medis. CNN terdiri dari lapisan konvolusional yang berfungsi mendeteksi fitur lokal pada citra, lapisan pooling yang mengurangi dimensi citra, serta lapisan fully connected untuk melakukan klasifikasi. Keunggulan CNN dalam menangani pengenalan pola yang kompleks menjadikannya model yang ideal untuk aplikasi medis, termasuk deteksi penyakit berbasis citra medis (Huang et al., 2021; Rajaraman et al., 2022).

Dalam konteks klasifikasi pneumonia, CNN telah terbukti efektif, terutama pada citra X-ray dada. Li et al. (2021) mengembangkan model CNN untuk mendeteksi pneumonia menggunakan citra X-ray dada, dan hasilnya menunjukkan bahwa model ini dapat mengidentifikasi pneumonia dengan akurasi yang lebih tinggi dibandingkan dengan metode konvensional. Dengan menggunakan arsitektur CNN yang dilatih untuk mengenali pola pada citra X-ray, model ini mampu mendeteksi pneumonia baik yang bersifat bakterial maupun viral dengan tingkat akurasi yang signifikan. Penelitian ini mengkonfirmasi bahwa CNN memiliki potensi besar dalam otomatisasi diagnosa pneumonia, yang dapat meningkatkan efisiensi dan akurasi diagnosis medis.

Rajaraman et al. (2022) juga melakukan penelitian serupa dengan mengimplementasikan CNN pada dataset X-ray dada untuk klasifikasi pneumonia. Dalam studi ini, mereka menguji berbagai arsitektur CNN, termasuk VGG-16 dan ResNet, yang menunjukkan bahwa model CNN yang lebih dalam seperti VGG-16 mampu memberikan akurasi yang lebih baik dibandingkan dengan model CNN yang lebih sederhana. Penggunaan arsitektur yang lebih dalam meningkatkan kemampuan model untuk mengenali pola yang lebih kompleks pada citra X-ray, yang sangat penting dalam mendeteksi pneumonia dan membedakannya dari kondisi paru-paru lainnya.

Penelitian oleh Xie et al. (2021) menguji penggunaan CNN pada citra CT scan dada untuk mendeteksi pneumonia, yang memberikan perspektif baru dalam aplikasi CNN pada citra medis selain X-ray. Penelitian ini menemukan bahwa CNN yang dilatih dengan baik pada citra CT scan dapat mencapai akurasi yang sebanding dengan dokter ahli dalam mendeteksi pneumonia. Lebih lanjut, teknik augmentasi data digunakan untuk mengatasi kekurangan data dalam citra medis, dan model CNN yang lebih kompleks terbukti meningkatkan akurasi diagnosis. Temuan ini menunjukkan bahwa CNN dapat diterapkan tidak hanya pada X-ray, tetapi juga pada citra medis lainnya seperti CT scan, memperluas aplikasinya dalam deteksi pneumonia.

Meskipun CNN menunjukkan hasil yang menjanjikan dalam klasifikasi pneumonia, salah satu tantangan utama yang dihadapi adalah kebutuhan akan komputasi yang sangat tinggi dan waktu pelatihan yang lama. Hal ini bisa menjadi kendala dalam penerapannya di lingkungan rumah sakit dengan keterbatasan sumber daya. Untuk mengatasi hal ini, beberapa penelitian mengusulkan penggunaan metode pembelajaran cepat, seperti Extreme Learning Machine (ELM), untuk menggantikan lapisan klasifikasi tradisional dalam CNN. Huang et al. (2006) menyarankan bahwa penggunaan ELM dapat mempercepat proses pelatihan tanpa mengorbankan akurasi model, memberikan solusi yang efisien dalam pengklasifikasian citra medis.
Secara keseluruhan, meskipun CNN terbukti sangat efektif dalam klasifikasi pneumonia, tantangan terkait dengan waktu pelatihan dan komputasi yang tinggi tetap menjadi perhatian. Oleh karena itu, penelitian lebih lanjut diperlukan untuk mengembangkan solusi seperti integrasi CNN dengan metode pembelajaran cepat seperti ELM, untuk meningkatkan efisiensi, akurasi, dan penerapan praktis dalam diagnosis otomatis pneumonia.

![image](https://github.com/user-attachments/assets/0e28889c-21f1-4a2f-b303-79343f9f5ccb)

Gambar 2.3 Convolutional Neural Networks (CNN)

### 2.4 	VGG-16
VGG-16 merupakan salah satu arsitektur Convolutional Neural Network (CNN) yang terkenal karena kesederhanaan dan efektivitasnya dalam tugas klasifikasi citra medis, termasuk dalam klasifikasi penyakit pneumonia. Model ini dikembangkan oleh Simonyan dan Zisserman (2014) dan memiliki 16 lapisan yang terdiri dari lapisan konvolusional dan lapisan fully connected. Sejak diperkenalkan, VGG-16 telah digunakan secara luas dalam berbagai aplikasi pengolahan citra medis, termasuk dalam mendeteksi penyakit seperti pneumonia, dengan hasil yang menjanjikan (Simonyan & Zisserman, 2014).

Dalam penelitian terbaru oleh Rajaraman et al. (2022), VGG-16 diterapkan untuk mengklasifikasikan pneumonia pada citra X-ray dada. Hasil penelitian menunjukkan bahwa VGG-16 dapat mendeteksi pneumonia dengan akurasi yang lebih tinggi dibandingkan model CNN lainnya yang lebih sederhana. Penelitian ini juga menekankan bahwa meskipun VGG-16 memiliki arsitektur yang lebih dalam, model ini tetap mampu memberikan hasil yang efektif dalam mengidentifikasi pola-pola pada citra X-ray yang berkaitan dengan gejala pneumonia. Hasil penelitian ini menyoroti kemampuan VGG-16 dalam menangani kompleksitas citra medis dan menghasilkan deteksi yang lebih akurat (Rajaraman et al., 2022).

Selain itu, Zhang et al. (2020) menggunakan VGG-16 untuk mendiagnosis pneumonia dari citra CT scan dada. Mereka menemukan bahwa dengan mengadaptasi arsitektur VGG-16 untuk citra CT scan, model ini dapat menghasilkan akurasi yang sangat baik dalam mendeteksi pneumonia. Penelitian ini juga menyoroti pentingnya teknik augmentasi data untuk memperbaiki performa model, terutama ketika dataset yang digunakan terbatas. Dengan menambahkan variasi pada data pelatihan, VGG-16 dapat lebih baik dalam mengenali pola-pola yang ada pada citra medis yang beragam (Zhang et al., 2020).

Selain aplikasi dalam X-ray dan CT scan, beberapa penelitian juga menunjukkan penerapan VGG-16 untuk klasifikasi pneumonia pada citra medis lainnya, seperti foto-foto medis yang diambil menggunakan teknologi ultrasound dan MRI. Penelitian oleh Xie et al. (2021) menunjukkan bahwa meskipun VGG-16 lebih dikenal untuk X-ray dan CT scan, model ini juga dapat disesuaikan untuk menganalisis citra medis dari modality yang berbeda. Mereka mengadaptasi VGG-16 dengan beberapa modifikasi dan mendapatkan hasil yang kompetitif dalam mendeteksi pneumonia pada citra MRI dada. Hal ini menunjukkan bahwa VGG-16 merupakan model yang fleksibel dan dapat diterapkan di berbagai jenis citra medis (Xie et al., 2021).

Namun, meskipun VGG-16 telah terbukti sangat efektif dalam klasifikasi pneumonia, salah satu tantangan terbesar yang dihadapi adalah komputasi yang tinggi dan waktu pelatihan yang lama. Untuk mengatasi tantangan ini, beberapa peneliti telah mencoba mengoptimalkan model VGG-16 dengan menggunakan metode transfer learning dan fine-tuning pada dataset yang lebih kecil. Misalnya, Yadav dan Jadhav (2019) mengadopsi teknik transfer learning dengan menggunakan VGG-16 yang telah dilatih pada dataset ImageNet dan melatihnya lebih lanjut pada dataset pneumonia yang lebih kecil. Mereka berhasil meningkatkan akurasi model sambil mengurangi waktu pelatihan secara signifikan, yang menunjukkan bahwa teknik ini dapat meningkatkan efisiensi pelatihan tanpa mengorbankan performa model.

Secara keseluruhan, VGG-16 terbukti sebagai model CNN yang efektif dan sangat berguna dalam klasifikasi citra medis, terutama untuk deteksi pneumonia. Namun, tantangan yang terkait dengan sumber daya komputasi dan waktu pelatihan yang lama tetap menjadi isu yang perlu diperhatikan. Oleh karena itu, terus dilakukan penelitian untuk meningkatkan efisiensi pelatihan dan optimasi model, dengan mempertimbangkan penggunaan teknik seperti transfer learning dan augmentasi data untuk meningkatkan kinerja VGG-16 dalam aplikasi dunia nyata, khususnya dalam diagnosis otomatis penyakit seperti pneumonia.
 
![image](https://github.com/user-attachments/assets/f9403727-4cfa-4af6-80e6-d5a024c8baa0)

Gambar 2.4 Arsitektur Model VGG16 (Swastika, 2020)

Input untuk arsitektur model VGG-16 terdiri dari gambar berukuran 224x224 piksel dengan 3 saluran warna (RGB). Pada lapisan konvolusi pertama (conv-1), terdapat 64 filter dengan ukuran kernel 3x3, diikuti oleh fungsi aktivasi ReLU dan operasi max pooling untuk mengurangi dimensi spasial. Selanjutnya, pada konvolusi kedua terdapat 128 filter, pada konvolusi ketiga terdapat 256 filter, dan pada konvolusi keempat serta kelima masing-masing menggunakan 512 filter. Di akhir jaringan, terdapat tiga lapisan fully connected (fc) dengan fungsi aktivasi ReLU, yang diikuti oleh lapisan output. (Swastika, 2020).

### 2.5	Extreme Learning Machine (ELM)
Extreme Learning Machine (ELM) adalah metode pembelajaran mesin yang dikenal karena kecepatan pelatihan yang tinggi dan kemampuan generalisasi yang baik. Dalam beberapa tahun terakhir, ELM telah diterapkan secara luas dalam klasifikasi citra medis, termasuk deteksi penyakit seperti pneumonia.
Salah satu penelitian yang relevan adalah oleh Putu Prima Winangun dkk. (2020), yang mengembangkan sistem pakar berbasis ELM dengan kernel linear untuk mengklasifikasikan kelainan paru-paru. Penelitian ini menunjukkan bahwa ELM dapat digunakan sebagai opini kedua untuk mendukung diagnosis dari pakar medis, dengan kecepatan pelatihan yang signifikan lebih cepat dibandingkan metode jaringan saraf tiruan konvensional. 
Selain itu, penelitian oleh Tarek Berghout (2020) menyediakan implementasi ELM untuk klasifikasi dan regresi, yang dapat digunakan dalam berbagai aplikasi, termasuk klasifikasi citra medis. Implementasi ini memungkinkan pengguna untuk melatih jaringan saraf tiruan dengan satu lapisan tersembunyi untuk tugas klasifikasi dan regresi, yang dapat diterapkan dalam analisis citra medis. 
Meskipun demikian, penelitian yang secara spesifik membahas penerapan ELM dalam klasifikasi pneumonia pada citra medis dalam lima tahun terakhir terbatas. Oleh karena itu, penelitian lebih lanjut diperlukan untuk mengeksplorasi potensi ELM dalam deteksi otomatis pneumonia menggunakan citra medis.

### 2.6	Variasi Input
Variasi input citra memainkan peran penting dalam meningkatkan kinerja model klasifikasi citra. Penyesuaian ukuran, resolusi, dan komposisi warna citra dapat memengaruhi efektivitas model dalam mengenali pola dan fitur penting.

Dalam penelitian yang dilakukan oleh Siti Nurul Hidayah dkk. (2022), dilakukan pengujian terhadap variasi ukuran citra input untuk mengamati dampaknya terhadap kinerja model Convolutional Neural Network (CNN) dalam klasifikasi pneumonia. Hasil penelitian menunjukkan bahwa variasi ukuran citra input memengaruhi akurasi model, dengan ukuran citra tertentu memberikan hasil yang lebih optimal dalam mendeteksi pneumonia. 

Selain itu, penelitian oleh Muhammad Rizki dkk. (2021) membahas analisis kemampuan klasifikasi citra berbasis objek dengan menggunakan segmentasi citra. Penelitian ini menunjukkan bahwa pemilihan saluran input citra yang tepat, seperti kombinasi saluran warna dan tekstur, dapat meningkatkan akurasi klasifikasi citra penutup lahan. 

Lebih lanjut, studi oleh Muhammad Rizki dkk. (2013) mengkaji klasifikasi citra satelit menggunakan kombinasi fitur warna dan tekstur. Penelitian ini menyoroti pentingnya pemilihan fitur yang sesuai dalam citra satelit untuk meningkatkan akurasi klasifikasi, dengan mempertimbangkan variasi dalam fitur warna dan tekstur citra. 

Secara keseluruhan, variasi input citra, termasuk ukuran, resolusi, dan teknik augmentasi, memiliki dampak signifikan terhadap kinerja model klasifikasi citra. Penelitian-penelitian tersebut menunjukkan bahwa pemilihan dan penyesuaian variasi input citra yang tepat dapat meningkatkan akurasi dan efektivitas model dalam berbagai aplikasi klasifikasi citra.

### 2.7	Komposisi Warna HSV
Penggunaan ruang warna HSV (Hue, Saturation, Value) dalam klasifikasi citra telah menjadi fokus berbagai penelitian dalam beberapa tahun terakhir. Ruang warna ini dipilih karena kemampuannya dalam memisahkan informasi intensitas (Value) dari informasi warna (Hue dan Saturation), yang lebih sesuai dengan persepsi visual manusia dibandingkan dengan ruang warna RGB.

Salah satu penelitian terbaru oleh Mallick et al. (2024) mengusulkan pendekatan baru untuk klasifikasi citra histopatologi kanker payudara dengan menggunakan fitur dari berbagai ruang warna, termasuk RGB dan HSV. Studi ini menunjukkan bahwa penggabungan fitur dari ruang warna yang berbeda dapat meningkatkan akurasi klasifikasi, dengan HSV memberikan kontribusi signifikan dalam representasi fitur warna. 

Selain itu, penelitian oleh Wang et al. (2021) mengembangkan UIEC^2-Net, sebuah jaringan konvolusional yang memanfaatkan ruang warna RGB dan HSV untuk peningkatan citra bawah air. Pendekatan ini menunjukkan bahwa integrasi informasi dari kedua ruang warna dapat meningkatkan kualitas citra yang dihasilkan, yang pada gilirannya dapat meningkatkan kinerja model klasifikasi yang menggunakan citra tersebut. 

Lebih lanjut, studi oleh Tushar (2018) menerapkan metode segmentasi GrabCut dalam ruang warna HSV untuk segmentasi lesi kulit secara otomatis. Pendekatan ini berhasil mencapai indeks Jaccard rata-rata sebesar 0,71 pada dataset ISIC 2017, menunjukkan efektivitas ruang warna HSV dalam meningkatkan akurasi segmentasi citra medis. 

Secara keseluruhan, penerapan ruang warna HSV dalam klasifikasi citra telah terbukti efektif dalam berbagai domain, termasuk segmentasi lesi kulit dan pengenalan warna objek. Kemampuan HSV untuk memisahkan komponen warna dan intensitas memungkinkan model pembelajaran mesin untuk lebih mudah mengenali pola dan fitur penting, sehingga meningkatkan akurasi dan kinerja dalam tugas klasifikasi citra.

### 2.8	Komposisi Warna RGB
Ruang warna RGB (Red, Green, Blue) adalah salah satu model warna yang paling umum digunakan dalam pengolahan citra digital. Dalam model ini, setiap warna direpresentasikan sebagai kombinasi dari tiga kanal warna primer: merah (R), hijau (G), dan biru (B). Nilai intensitas setiap kanal biasanya berada dalam rentang 0 hingga 255, yang memungkinkan representasi warna dalam jutaan kombinasi unik. Model RGB banyak digunakan karena kesederhanaannya dan kompatibilitasnya dengan berbagai perangkat seperti kamera digital dan monitor.

Penelitian terbaru menunjukkan bahwa penggunaan ruang warna RGB dalam tugas klasifikasi citra memberikan hasil yang sangat baik, terutama dalam aplikasi medis dan non-medis. Misalnya, Li et al. (2021) memanfaatkan citra X-ray dalam format RGB untuk mendeteksi pneumonia menggunakan model CNN. Mereka menunjukkan bahwa preprocessing seperti normalisasi nilai intensitas RGB dapat meningkatkan akurasi model hingga lebih dari 90%. Penelitian ini menekankan pentingnya penyesuaian nilai intensitas untuk mengurangi pengaruh pencahayaan yang tidak seragam pada data mentah.

Selain itu, Nguyen et al. (2023) mengaplikasikan model warna RGB untuk tugas klasifikasi kanker kulit. Dalam penelitian ini, data RGB diolah dengan teknik histogram equalization, yang mampu menyeimbangkan intensitas warna dan menghasilkan akurasi model hingga 94%. Penelitian ini juga menemukan bahwa ruang warna RGB sangat cocok untuk tugas klasifikasi dengan tekstur kompleks, seperti citra kulit manusia, karena mampu mempertahankan detail warna yang esensial.

Meskipun RGB memiliki keunggulan dalam representasi warna yang detail, ada beberapa tantangan dalam penggunaannya, seperti sensitivitas terhadap perubahan pencahayaan. Zhou et al. (2022) menunjukkan bahwa augmentasi data, seperti rotasi dan perubahan nilai intensitas RGB, dapat mengatasi permasalahan ini. Penelitian mereka menggunakan citra satelit dalam format RGB untuk klasifikasi area geografis, dengan hasil peningkatan akurasi sebesar 15% setelah menerapkan augmentasi data berbasis RGB.

Sharma et al. (2020) juga membahas penggunaan RGB untuk klasifikasi citra makanan. Mereka menemukan bahwa meskipun RGB efektif, kinerjanya dapat ditingkatkan dengan menggabungkan informasi dari ruang warna lain, seperti HSV. Studi ini menunjukkan bahwa model berbasis RGB sering kali memerlukan preprocessing tambahan untuk menangani pencahayaan yang bervariasi.

Secara keseluruhan, RGB tetap menjadi pilihan utama dalam pengolahan citra digital, terutama dalam tugas klasifikasi. Namun, efektivitasnya sangat bergantung pada metode preprocessing yang diterapkan, seperti normalisasi atau augmentasi data. Inovasi terbaru dalam kombinasi RGB dengan ruang warna lain juga menunjukkan potensi untuk meningkatkan performa model lebih lanjut.

### 2.9	Konversi RGB ke HSV dalam Klasifikasi Citra
Ruang warna RGB (Red, Green, Blue) sering digunakan dalam berbagai aplikasi pengolahan citra karena fleksibilitasnya. Namun, dalam beberapa kasus, terutama dalam analisis citra yang membutuhkan interpretasi warna berdasarkan hue (warna), saturation (kejenuhan), dan value (kecerahan), ruang warna HSV (Hue, Saturation, Value) menjadi lebih efektif. Konversi dari RGB ke HSV memungkinkan representasi warna yang lebih mendekati persepsi manusia, terutama dalam pengklasifikasian pola warna pada citra.

Penelitian terbaru oleh Li et al. (2023) menunjukkan bahwa penggunaan ruang warna HSV pada citra X-ray dada dapat meningkatkan akurasi deteksi pneumonia hingga 5% dibandingkan dengan ruang warna RGB. Dalam penelitian ini, citra X-ray yang awalnya dalam format RGB dikonversi ke HSV untuk lebih menonjolkan area kontras tinggi, seperti pada paru-paru yang terkena pneumonia. Proses konversi membantu dalam menyesuaikan intensitas warna yang tidak seragam akibat pencahayaan yang berbeda.

Dalam aplikasi lain, Nguyen et al. (2022) mengaplikasikan model warna HSV pada klasifikasi penyakit kulit. Mereka menggabungkan data dari ruang warna HSV dengan RGB untuk menciptakan fitur gabungan yang lebih informatif. Konversi ini memungkinkan model untuk mengenali pola warna yang lebih halus, terutama dalam kasus di mana perubahan warna kulit menjadi indikator utama penyakit. Dengan teknik ini, akurasi model meningkat sebesar 8% dibandingkan dengan hanya menggunakan RGB.

Proses konversi RGB ke HSV juga memberikan manfaat dalam meningkatkan robusta terhadap perubahan pencahayaan. Dalam penelitian oleh Sharma et al. (2023), citra makanan yang awalnya dalam format RGB dikonversi ke HSV untuk mengenali jenis makanan berdasarkan warna dan tekstur. Dengan mengisolasi informasi hue, model berhasil mengabaikan perubahan pencahayaan yang sering memengaruhi keakuratan klasifikasi pada ruang warna RGB. Hasilnya, akurasi model meningkat sebesar 10%.

Penelitian oleh Zhou et al. (2021) pada citra satelit menunjukkan bahwa HSV lebih unggul dalam mendeteksi objek tertentu, seperti area hijau atau lahan basah, dibandingkan dengan RGB. Dalam penelitian ini, hue digunakan untuk mendeteksi warna dominan, sementara value membantu mengidentifikasi area dengan intensitas tinggi. Teknik ini memungkinkan pengklasifikasian objek dengan presisi lebih tinggi meskipun terdapat variasi pencahayaan.

Secara keseluruhan, konversi dari RGB ke HSV memberikan keuntungan signifikan dalam tugas klasifikasi citra, terutama ketika pola warna menjadi faktor utama. Selain itu, integrasi antara ruang warna RGB dan HSV dapat memberikan informasi tambahan yang meningkatkan akurasi model. Penggunaan HSV yang lebih luas dalam klasifikasi citra medis menunjukkan potensi besar untuk mengoptimalkan analisis data berbasis citra.

### 2.10	Integrasi Convolutional Neural Network (CNN) dengan Extreme Learning Machine
(ELM)Integrasi Convolutional Neural Network (CNN) dengan Extreme Learning Machine (ELM) telah menarik perhatian banyak peneliti, terutama dalam aplikasi pengolahan citra medis. CNN dikenal dengan kemampuannya dalam mengekstraksi fitur kompleks dari citra melalui lapisan konvolusi dan pooling, sementara ELM berfungsi sebagai pengklasifikasi yang cepat dan efisien dengan menggunakan lapisan fully connected untuk klasifikasi (Huang et al., 2006). Pendekatan hibrida ini memberikan solusi untuk mengatasi tantangan dalam klasifikasi citra medis yang memerlukan akurasi tinggi serta efisiensi waktu dan sumber daya.

Salah satu penelitian yang relevan adalah studi oleh Ahmad Hasby Bik, Fetty Tri Anggraeny, dan Eva Yulia Puspaningrum yang dipublikasikan pada Juni 2024. Dalam penelitian ini, model hibrida CNN-ELM digunakan untuk mengklasifikasikan gambar CT penyakit ginjal. Hasil eksperimen menunjukkan bahwa penggunaan fungsi aktivasi ReLU dalam ELM menghasilkan akurasi tertinggi sebesar 99,63%, sedangkan fungsi Tanh hanya mencapai 84,19%. Penelitian ini menunjukkan bahwa pemilihan fungsi aktivasi yang tepat dalam ELM dapat meningkatkan kinerja model hibrida CNN-ELM dalam mendiagnosis penyakit ginjal (Bik et al., 2024).

Penelitian lain oleh D. Gunawan dan H. Setiawan, yang dipublikasikan pada Desember 2022, meskipun tidak langsung membahas integrasi CNN dengan ELM, memberikan wawasan tentang efektivitas CNN dalam analisis citra medis. Dalam studi ini, CNN digunakan untuk berbagai aplikasi dalam citra medis, seperti klasifikasi, deteksi, segmentasi, dan peningkatan citra. Keberhasilan CNN dalam bidang ini menunjukkan potensi besar teknologi ini untuk membantu diagnosis dan deteksi otomatis dalam praktik medis, meskipun tantangan besar terkait dengan kebutuhan komputasi tetap ada (Gunawan & Setiawan, 2022).

Secara keseluruhan, integrasi CNN dengan ELM memberikan pendekatan yang menjanjikan dalam klasifikasi citra medis. Pemilihan fungsi aktivasi yang tepat dalam ELM serta pemahaman tentang arsitektur CNN yang tepat menjadi kunci utama dalam mengoptimalkan kinerja model hibrida ini. Penelitian yang ada menunjukkan bahwa kombinasi kedua teknik ini dapat mengurangi waktu pelatihan tanpa mengorbankan akurasi, memberikan kontribusi signifikan dalam pengembangan model untuk diagnosis penyakit yang lebih efisien dan akurat (Bik et al., 2024; Gunawan & Setiawan, 2022).

### 2.11	Pengolahan Citra Digital
Pengolahan citra digital adalah cabang ilmu komputer yang berfokus pada manipulasi citra menggunakan algoritma dan teknik komputer untuk meningkatkan kualitas atau mengubah informasi dalam citra. Proses ini mencakup berbagai teknik seperti perbaikan kualitas citra, ekstraksi fitur, segmentasi, dan klasifikasi. Dengan kemajuan teknologi, pengolahan citra digital telah banyak digunakan dalam berbagai aplikasi seperti pengenalan wajah, deteksi objek, dan diagnosa medis menggunakan citra medis.

Dalam beberapa tahun terakhir, penerapan deep learning dalam pengolahan citra digital telah menunjukkan kemajuan yang signifikan. Deep learning, khususnya Convolutional Neural Networks (CNN), telah digunakan untuk berbagai tugas seperti klasifikasi objek, restorasi citra, deteksi anomali, dan segmentasi. CNN memiliki kemampuan untuk mengekstrak fitur hierarkis dari citra, mulai dari fitur-fitur rendah seperti garis dan sudut hingga fitur-fitur yang lebih kompleks seperti bentuk dan tekstur. Selama proses pelatihan, CNN belajar untuk menyesuaikan bobot-bobot mereka agar dapat menghasilkan representasi yang optimal dari citra pelatihan. Ini dilakukan dengan meminimalkan fungsi kerugian yang mengukur perbedaan antara prediksi yang dibuat oleh model dan label yang sebenarnya dari citra. 

Selain itu, pengolahan citra digital juga diterapkan dalam analisis data visual. Teknologi pemrosesan gambar digunakan untuk menganalisis data visual, dengan menekankan metode dan alat yang digunakan dalam proses tersebut. Penerapan teknologi ini memungkinkan analisis data visual yang lebih efektif dan efisien, yang dapat digunakan dalam berbagai bidang seperti pengawasan, keamanan, dan analisis data ilmiah. 

Dalam bidang medis, pengolahan citra digital digunakan untuk analisis elektrokardiogram (EKG). Pengolahan citra digital EKG memungkinkan analisis yang lebih cepat dan akurat, yang penting untuk diagnosis dan pemantauan kondisi jantung pasien. Dengan menggunakan teknik pengolahan citra, informasi dari EKG dapat diekstraksi dan dianalisis secara otomatis, mengurangi ketergantungan pada interpretasi manual dan meningkatkan efisiensi dalam praktik medis. 

Secara keseluruhan, pengolahan citra digital telah menjadi bidang yang sangat penting dengan berbagai aplikasi di berbagai sektor. Kemajuan dalam teknologi deep learning dan penerapannya dalam pengolahan citra digital telah membuka peluang baru untuk analisis dan pemrosesan citra yang lebih efektif dan efisien.

### 2.12	Pembelajaran Mendalam (Deep Learning)
Deep learning adalah sub-bidang dari machine learning yang menggunakan model jaringan saraf tiruan dengan banyak lapisan (layer) untuk memproses data dan belajar representasi dari data secara otomatis. Pendekatan ini telah berhasil diterapkan di berbagai bidang, seperti pengenalan gambar, pemrosesan bahasa alami, dan pengenalan suara. Salah satu keunggulan utama deep learning adalah kemampuannya dalam mengekstraksi fitur secara otomatis dari data mentah, tanpa memerlukan fitur manual yang diajukan sebelumnya oleh pengembang. Model deep learning, seperti Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), dan Generative Adversarial Networks (GAN), telah mengubah cara kita memandang dan memanfaatkan data dalam berbagai aplikasi.

Beberapa penelitian menunjukkan bagaimana deep learning dapat digunakan dalam aplikasi kesehatan, misalnya dalam pengolahan citra medis. Model CNN, misalnya, telah digunakan dalam tugas-tugas klasifikasi citra medis, seperti pengenalan tumor pada citra X-ray atau MRI (Liu et al., 2021). Deep learning juga digunakan untuk memprediksi kondisi pasien berdasarkan data medis yang lebih kompleks, meningkatkan efisiensi diagnosis otomatis. Sebagai contoh, dalam studi oleh Wang et al. (2020), deep learning diterapkan untuk menganalisis citra radiologi, dengan hasil yang mengesankan dalam meningkatkan akurasi dan mengurangi kesalahan manusia dalam proses diagnosa.

Penggunaan deep learning di bidang pengenalan pola juga semakin meluas, dengan arsitektur jaringan saraf dalam yang mampu menangani dataset besar dan kompleks dengan efisien. Dalam domain pengenalan wajah, model deep learning telah mampu mencapai hasil yang hampir setara dengan kemampuan manusia dalam mengidentifikasi individu berdasarkan citra wajah (Zhang et al., 2022). Hal ini menunjukkan bahwa teknologi deep learning memiliki potensi besar dalam memecahkan masalah klasifikasi dan prediksi yang sebelumnya sulit untuk dicapai dengan teknik konvensional.

Meskipun deep learning memiliki keunggulan yang signifikan, tantangan utama yang dihadapi adalah kebutuhan akan data dalam jumlah besar dan sumber daya komputasi yang tinggi. Namun, dengan kemajuan hardware seperti GPU dan peningkatan teknik optimisasi, banyak penelitian yang menunjukkan bahwa hambatan ini semakin dapat diatasi. Dalam penelitian oleh Chen et al. (2023), teknik transfer learning dan penggunaan dataset pre-trained model terbukti efektif dalam mengurangi ketergantungan pada data pelatihan besar dan mempercepat proses pelatihan.

Secara keseluruhan, deep learning terus berkembang dan menunjukkan hasil yang menjanjikan di berbagai bidang. Dengan kemajuan teknologi dan penelitian, deep learning akan semakin banyak digunakan dalam berbagai aplikasi, termasuk dalam bidang kesehatan, transportasi, dan lain-lain.

### 2.13	Pembelajaran Mesin (Machine Learning)
Machine Learning (ML) adalah cabang dari kecerdasan buatan yang berfokus pada pengembangan algoritma dan model statistik yang memungkinkan komputer untuk "belajar" dari data tanpa pemrograman eksplisit. Dalam pengolahan data, machine learning digunakan untuk mengenali pola, membuat prediksi, dan memecahkan masalah yang kompleks dengan mengandalkan data yang ada. Berbagai jenis algoritma ML, seperti Supervised Learning, Unsupervised Learning, dan Reinforcement Learning, masing-masing memiliki aplikasi spesifik tergantung pada jenis masalah yang ingin diselesaikan.

Dalam konteks pengolahan citra medis, machine learning telah digunakan untuk berbagai tugas, termasuk deteksi penyakit, segmentasi citra, dan klasifikasi. Misalnya, dalam penelitian oleh Jain et al. (2021), sebuah model supervised learning berbasis Random Forest digunakan untuk mendiagnosis kanker paru-paru berdasarkan citra X-ray, dengan hasil akurasi yang sangat baik mencapai 94%. Teknik ini menunjukkan bagaimana algoritma ML dapat mengidentifikasi pola yang berkaitan dengan kelainan dalam citra medis, membantu dalam proses diagnosa yang lebih cepat dan akurat.

Selain itu, pada metode unsupervised learning, algoritma seperti K-Means clustering atau Principal Component Analysis (PCA) banyak digunakan untuk segmentasi citra dan pengelompokan data tanpa memerlukan label yang sudah ditentukan sebelumnya. Menurut penelitian oleh Park et al. (2020), K-Means clustering diterapkan untuk mengelompokkan citra jaringan tumor pada mamografi, yang memungkinkan identifikasi area-area yang mencurigakan untuk diagnosis lanjutan.

Dalam bidang pengenalan pola, machine learning semakin banyak diterapkan dalam sistem rekomendasi, pemrosesan bahasa alami, dan kendaraan otonom. Sebagai contoh, penelitian oleh Smith et al. (2022) menggali potensi Support Vector Machines (SVM) dalam mendeteksi perilaku mencurigakan dari data CCTV untuk aplikasi keamanan. Penelitian ini menunjukkan bagaimana SVM, sebagai salah satu metode dari supervised learning, sangat efektif dalam klasifikasi berdasarkan fitur-fitur tertentu yang diekstraksi dari citra atau data lain.

Meski demikian, tantangan utama dalam machine learning adalah kualitas dan kuantitas data yang diperlukan untuk pelatihan model yang baik. Seiring dengan meningkatnya ukuran dataset dan kompleksitas masalah, kebutuhan akan sumber daya komputasi juga meningkat. Penelitian oleh Yang et al. (2023) menunjukkan bahwa penggunaan Deep Learning dalam conjunction dengan machine learning dapat mengatasi masalah ini dengan melakukan otomatisasi ekstraksi fitur dan meningkatkan akurasi prediksi, meskipun membutuhkan data dalam jumlah besar untuk pelatihan.

Secara keseluruhan, machine learning memiliki potensi besar untuk diterapkan di berbagai bidang, termasuk kesehatan, keuangan, dan teknologi, dengan berbagai jenis algoritma yang dapat disesuaikan dengan kebutuhan spesifik dari masalah yang dihadapi.

### 2.14	Feature Learning
Feature Learning adalah teknik dalam machine learning yang bertujuan untuk secara otomatis mengekstrak fitur-fitur yang relevan dari data mentah untuk digunakan dalam model prediksi atau klasifikasi. Berbeda dengan pendekatan tradisional yang mengandalkan ekstraksi fitur manual, feature learning memungkinkan model untuk "belajar" fitur-fitur terbaik dari data dengan sedikit atau tanpa intervensi manusia. Metode ini sangat penting dalam aplikasi yang melibatkan data tidak terstruktur seperti citra, suara, dan teks.

Dalam konteks Deep Learning, feature learning menjadi salah satu aspek penting, terutama dalam arsitektur jaringan saraf yang mendalam seperti Convolutional Neural Networks (CNN), yang dapat mengekstraksi fitur-fitur yang lebih kompleks secara hierarkis. Penelitian oleh Liu et al. (2020) menunjukkan bagaimana CNN dapat digunakan untuk feature learning dalam klasifikasi citra medis, seperti deteksi tumor dalam citra X-ray dan MRI. Dalam studi ini, CNN tidak hanya mengidentifikasi pola dasar seperti tepi atau bentuk, tetapi juga fitur tingkat tinggi yang lebih relevan untuk diagnosis penyakit.

Selain CNN, ada beberapa teknik lain dalam feature learning yang juga banyak digunakan, seperti Autoencoders dan Restricted Boltzmann Machines (RBM). Penelitian oleh Zhang et al. (2021) menunjukkan bahwa autoencoders, yang merupakan jenis jaringan saraf yang dilatih untuk memetakan input ke representasi yang lebih sederhana, dapat digunakan untuk mereduksi dimensi data citra medis sambil mempertahankan fitur-fitur penting yang mendukung klasifikasi penyakit.

Penerapan feature learning juga terbukti efektif dalam pengenalan pola suara dan teks. Dalam pengenalan suara, misalnya, metode feature learning digunakan untuk mengekstraksi fitur penting dari sinyal suara yang kemudian digunakan untuk pengklasifikasian atau transkripsi suara ke teks. Penelitian oleh Chen et al. (2022) mengungkapkan bagaimana teknik feature learning dapat meningkatkan akurasi sistem pengenalan suara otomatis dengan mengidentifikasi fitur-fitur suara yang lebih halus.

Salah satu tantangan utama dalam feature learning adalah kebutuhan untuk memiliki dataset yang cukup besar dan beragam, terutama pada aplikasi pengenalan citra medis, di mana fitur-fitur yang berguna mungkin sangat spesifik dan memerlukan pelatihan dengan data dalam jumlah besar. Penelitian oleh Lee et al. (2023) menyoroti pentingnya augmentasi data untuk meningkatkan kinerja model dalam mempelajari fitur yang lebih baik, terutama dalam konteks dataset yang terbatas.

Secara keseluruhan, feature learning adalah teknik yang sangat berguna dalam berbagai aplikasi machine learning, terutama untuk data yang tidak terstruktur. Kemampuan untuk mengekstraksi fitur-fitur penting secara otomatis memungkinkan model untuk memberikan prediksi yang lebih akurat dan efisien, terutama dalam domain yang kompleks seperti citra medis.

### 2.15	Classification
Classification adalah salah satu tugas utama dalam machine learning yang bertujuan untuk memetakan data input ke dalam kategori atau kelas tertentu. Proses ini memainkan peran penting dalam banyak aplikasi, seperti diagnosis medis, analisis citra, pengenalan pola, dan sistem rekomendasi. Dalam konteks machine learning, klasifikasi dilakukan dengan menggunakan algoritma yang dapat mempelajari pola-pola dalam data untuk menentukan kelas atau label yang tepat bagi data baru.

Dalam Supervised Learning, klasifikasi dilakukan dengan melatih model menggunakan dataset yang sudah dilabeli, yang memungkinkan model untuk belajar dan memetakan fitur-fitur data ke kelas yang sesuai. Salah satu algoritma yang sering digunakan dalam klasifikasi adalah Support Vector Machine (SVM). Penelitian oleh Zhang et al. (2022) mengungkapkan bahwa SVM dapat memberikan hasil yang sangat baik dalam mengklasifikasikan citra medis, terutama dalam mendeteksi penyakit seperti kanker atau pneumonia berdasarkan citra medis, seperti X-ray atau CT scan. Studi ini menyoroti efektivitas SVM dalam klasifikasi citra dengan menggunakan teknik ekstraksi fitur yang lebih canggih.

Selain SVM, Random Forest (RF) juga merupakan metode populer dalam klasifikasi. Penelitian oleh Gupta et al. (2023) menunjukkan bahwa Random Forest dapat digunakan untuk klasifikasi penyakit berbasis citra medis dengan memberikan akurasi yang tinggi. Dalam studi ini, RF berhasil mengklasifikasikan jenis-jenis kanker paru-paru dengan efisiensi tinggi setelah memanfaatkan teknik feature learning dan pemrosesan citra.

Metode lain yang sering digunakan adalah Deep Learning dengan jaringan saraf dalam seperti Convolutional Neural Networks (CNN). CNN telah banyak diterapkan dalam berbagai masalah klasifikasi citra, termasuk pengenalan penyakit dalam citra medis. Penelitian oleh Li et al. (2021) membahas penggunaan CNN untuk klasifikasi pneumonia dalam citra X-ray dada. Dalam penelitian ini, CNN digunakan untuk mengekstraksi fitur secara otomatis dari citra, yang kemudian diklasifikasikan untuk mendeteksi adanya pneumonia dengan akurasi yang sangat tinggi.

Klasifikasi juga dapat dilakukan dengan menggunakan model Extreme Learning Machine (ELM), yang dikenal dengan efisiensinya dalam hal komputasi. Penelitian oleh Wang et al. (2023) mengembangkan model hibrida yang mengintegrasikan CNN untuk feature extraction dan ELM untuk klasifikasi, yang menunjukkan peningkatan akurasi signifikan dibandingkan dengan menggunakan CNN atau ELM secara terpisah.

Secara keseluruhan, klasifikasi merupakan komponen penting dalam berbagai aplikasi machine learning, terutama di bidang pengolahan citra medis. Dengan kemajuan dalam metode seperti SVM, Random Forest, CNN, dan ELM, sistem klasifikasi dapat memberikan prediksi yang lebih akurat dan efisien, bahkan dalam domain yang sangat kompleks seperti diagnosis penyakit berbasis citra.

### 2.16	Citra Digital dalam Pengolahan Citra Medis
Citra digital merupakan representasi visual dari data yang telah diproses secara digital, yang menyimpan informasi visual dalam format numerik, seperti array pixel atau matriks. Dalam konteks pengolahan citra medis, citra digital digunakan untuk menggambarkan berbagai jenis gambar medis, seperti hasil X-ray, CT scan, atau MRI, yang kemudian diproses untuk mendiagnosis penyakit atau kondisi medis tertentu. Teknologi pengolahan citra digital telah berkembang pesat dengan menggunakan berbagai teknik seperti pemfilteran, segmentasi, deteksi fitur, dan klasifikasi (Jain et al., 2020).

Salah satu aplikasi penting dari citra digital adalah dalam bidang kesehatan, khususnya dalam membantu dokter dalam mendiagnosis penyakit dengan lebih cepat dan akurat. Penelitian menunjukkan bahwa citra digital dapat digunakan untuk memudahkan deteksi penyakit seperti kanker, pneumonia, dan penyakit jantung dengan cara yang lebih objektif dan efisien (Siddique et al., 2020). Selain itu, pengolahan citra medis juga dapat membantu dalam proses perencanaan pengobatan dan pemantauan pasien secara lebih akurat.

## BAB III
## METODE PENELITIAN

### 3.1 	Kerangka Penelitian
Kerangka penelitian ini dirancang untuk menganalisis performansi model VGG-16 yang terintegrasi dengan Extreme Learning Machine (ELM) dalam klasifikasi pneumonia.

![image](https://github.com/user-attachments/assets/24951f00-77ab-4396-a0b9-b978e2f9aba6)

Gambar 3.1 Flow Kerangka Penelitian

Kerangka penelitian terdiri dari lima tahapan utama, yaitu:

#### 3.1.1 Pengumpulan Data
Tahap ini mencakup pengumpulan dataset citra X-ray dada yang digunakan sebagai bahan utama dalam eksperimen. Dataset seperti Chest X-Ray Pneumonia Dataset yang tersedia di Kaggle sering digunakan dalam penelitian serupa (Kermany et al., 2018; Cohen et al., 2020). Dataset ini berisi citra X-ray dada yang sudah diklasifikasikan ke dalam dua kategori utama: pneumonia (bakterial/viral) dan normal.

#### 3.1.2 Preprocessing Data
Proses preprocessing mencakup beberapa langkah penting untuk memastikan data siap diproses oleh model. Langkah-langkahnya meliputi:
1.	Resizing: Citra diubah ukurannya menjadi 224x224 piksel sesuai kebutuhan arsitektur VGG-16 (Simonyan & Zisserman, 2015).
2.	Normalisasi: Setiap piksel citra dinormalisasi dalam rentang 0-1 agar model lebih stabil selama pelatihan (Wang et al., 2022).
3.	Transformasi HSV: Konversi citra RGB ke HSV dilakukan untuk mengeksplorasi pengaruh komposisi warna pada akurasi klasifikasi.
4.	Augmentasi Data: Teknik augmentasi seperti flipping, rotation, dan zoom digunakan untuk memperkaya dataset guna mengatasi masalah overfitting (Shorten & Khoshgoftaar, 2019).
   
#### 3.1.3 Ekstraksi Fitur Menggunakan VGG-16
VGG-16 adalah model deep learning yang dirancang untuk mengekstraksi fitur visual dari citra. Model ini menggunakan lapisan konvolusional untuk mendeteksi pola lokal dan lapisan pooling untuk mengurangi dimensi data. Dalam penelitian ini, bagian fully connected layer dari VGG-16 dimodifikasi agar dapat menghasilkan fitur yang lebih representatif untuk diklasifikasikan oleh ELM (Zhang et al., 2021).

#### 3.1.4 Klasifikasi Menggunakan ELM
ELM adalah metode pembelajaran mesin yang dikenal karena efisiensinya dalam pelatihan dan kemampuan klasifikasinya yang kuat. Dalam penelitian ini, fitur yang diekstraksi oleh VGG-16 diklasifikasikan menggunakan ELM. Metode ini mengurangi waktu pelatihan dan kebutuhan komputasi dibandingkan metode tradisional seperti softmax (Wen et al., 2023).

#### 3.1.5 Evaluasi Model
Model dievaluasi menggunakan metrik-metrik seperti akurasi, presisi, recall, dan F1-score. Evaluasi ini bertujuan untuk menilai performansi model dalam mengklasifikasikan citra X-ray ke dalam kategori pneumonia atau normal (Raja et al., 2022).

### 3.2 	Pengumpulan Data
Pengumpulan data merupakan tahap awal yang sangat penting dalam penelitian ini karena kualitas data secara langsung memengaruhi performansi model yang akan dikembangkan. Data yang digunakan dalam penelitian ini adalah citra X-ray dada yang berfungsi sebagai input utama untuk klasifikasi pneumonia. Berikut adalah rincian proses pengumpulan data penelitian:

#### Sumber Data
Dataset yang digunakan adalah dataset citra medis yang berasal dari sumber terbuka yang terpercaya, seperti:
1.	Chest X-Ray Images (Pneumonia): Dataset ini merupakan salah satu dataset populer yang tersedia di Kaggle. Dataset ini berisi ribuan gambar X-ray dada yang terbagi ke dalam tiga kategori: normal, pneumonia bakterial, dan pneumonia viral (Kermany et al., 2018).
2.	COVID-19 Image Data Collection: Dataset ini dikembangkan oleh Cohen et al. (2020) untuk membantu penelitian terkait penyakit paru-paru, termasuk pneumonia yang sering menjadi komplikasi. Dataset ini memuat citra X-ray pasien dengan COVID-19 dan pneumonia.
   
#### Karakteristik Dataset
Dataset yang digunakan memiliki karakteristik sebagai berikut:
•	Resolusi: Resolusi gambar beragam, namun untuk konsistensi dan kompatibilitas dengan model VGG-16, semua gambar diubah ukurannya menjadi 224x224 piksel.
•	Kelas: Dataset diklasifikasikan menjadi tiga kelas utama:
o	Normal: Gambar X-ray dada pasien tanpa kelainan.
o	Pneumonia Bakterial: Gambar pasien dengan pneumonia yang disebabkan oleh infeksi bakteri.
o	Pneumonia Viral: Gambar pasien dengan pneumonia akibat infeksi virus.
•	Jumlah Data: Dataset memiliki distribusi data yang bervariasi. Untuk mengatasi ketidakseimbangan data, teknik augmentasi digunakan untuk memperbanyak citra di kelas minoritas (Shorten & Khoshgoftaar, 2019).

#### Prosedur Pengumpulan Data
1.	Pengunduhan Data
Dataset diperoleh dari platform data terbuka seperti Kaggle dan repositori arXiv. Setiap dataset diverifikasi untuk memastikan label sesuai dengan kategori yang diinginkan.
2.	Pembersihan Data
Gambar-gambar yang buram, memiliki artefak, atau label yang salah diidentifikasi dan dihapus untuk menjaga kualitas data. Proses ini dilakukan dengan inspeksi manual dan algoritme deteksi artefak otomatis.
3.	Labeling Data
Setiap gambar dalam dataset sudah dilabeli oleh penyedia dataset. Namun, validasi ulang dilakukan untuk memastikan konsistensi data.

#### Keberagaman Dataset
Keberagaman data dalam hal resolusi, kualitas, dan perangkat X-ray yang digunakan menjadi perhatian utama. Dataset dengan variasi ini membantu memastikan bahwa model dapat bekerja dengan baik pada data yang berasal dari berbagai sumber (Gupta & Mittal, 2021).

#### Integritas Data
Dataset dipastikan memiliki kualitas tinggi dengan meminimalkan noise dan bias. Proses validasi manual oleh ahli radiologi dilakukan pada beberapa dataset untuk memastikan bahwa data mencerminkan kondisi medis yang sebenarnya.

### 3.3 	Preprocessing Data
Preprocessing data adalah langkah penting dalam penelitian ini untuk memastikan bahwa citra X-ray dada yang digunakan sebagai input memiliki kualitas dan format yang sesuai dengan kebutuhan model. Langkah ini mencakup berbagai teknik yang bertujuan untuk membersihkan, menormalkan, dan mempersiapkan data agar kompatibel dengan model VGG-16 yang diintegrasikan dengan ELM. Berikut adalah tahapan detail dari preprocessing data dalam penelitian ini:

#### 3.3.1. Resizing
Dataset citra X-ray memiliki resolusi yang bervariasi karena diambil dari berbagai sumber dan perangkat medis. Model VGG-16 mengharuskan input berupa citra dengan dimensi 224x224 piksel dan 3 channel warna (RGB). Oleh karena itu, setiap citra diubah ukurannya menggunakan algoritme interpolasi bilinear.

#### 3.3.2. Normalisasi
Citra X-ray memiliki intensitas piksel dengan rentang nilai antara 0 hingga 255. Untuk memastikan bahwa data cocok untuk model deep learning, nilai intensitas piksel dinormalisasi ke dalam rentang 0 hingga 1 dengan membagi setiap piksel dengan nilai maksimum (255). Normalisasi ini membantu mempercepat proses pelatihan model dengan menjaga skala data tetap konsisten.

#### 3.3.3. Konversi Warna
Dataset yang digunakan adalah citra X-ray yang pada dasarnya memiliki warna grayscale. Namun, model VGG-16 membutuhkan input dengan 3 channel (RGB). Untuk memenuhi kebutuhan ini, citra grayscale dikonversi ke citra RGB dengan metode duplikasi channel, sehingga setiap piksel memiliki nilai intensitas yang sama di ketiga channel.

#### 3.3.4. Augmentasi Data
Untuk mengatasi masalah keterbatasan data dan ketidakseimbangan kelas, dilakukan augmentasi data. Teknik augmentasi meliputi:
•	Rotasi: Citra diputar dalam sudut tertentu untuk meningkatkan keragaman orientasi.
•	Flip Horizontal dan Vertikal: Membalik citra secara horizontal dan vertikal untuk memperbanyak data.
•	Crop dan Zoom: Memotong atau memperbesar bagian tertentu dari citra untuk simulasi variasi skala.
Augmentasi membantu model mempelajari fitur yang lebih beragam, sehingga meningkatkan generalisasi.

#### 3.3.5. Reduksi Noise
Untuk memastikan bahwa model tidak terganggu oleh noise, citra diproses menggunakan filter Gaussian. Filter ini digunakan untuk menghaluskan citra dan mengurangi artefak yang mungkin muncul akibat kualitas rendah dari perangkat X-ray.

#### 3.3.6. Ekstraksi ROI (Region of Interest)
Pada beberapa kasus, bagian yang tidak relevan seperti tulang di luar area paru-paru dapat mengganggu proses klasifikasi. Oleh karena itu, dilakukan cropping atau segmentasi untuk mengekstrak ROI yang hanya mencakup area paru-paru. Segmentasi berbasis thresholding sederhana digunakan untuk membedakan antara jaringan paru-paru dan bagian lainnya.

#### 3.3.7. Penyeimbangan Data
Jika dataset memiliki jumlah data yang tidak seimbang antar kelas, seperti lebih banyak data untuk kelas normal dibandingkan pneumonia, dilakukan resampling atau oversampling untuk menyeimbangkan distribusi kelas. Teknik Synthetic Minority Oversampling Technique (SMOTE) sering digunakan dalam penelitian ini.

#### Kesimpulan
Tahapan preprocessing dalam penelitian ini memastikan bahwa data yang digunakan memenuhi standar kualitas tinggi dan kompatibel dengan model yang akan digunakan. Dengan teknik-teknik seperti resizing, normalisasi, augmentasi, dan ekstraksi ROI, model diharapkan dapat mempelajari fitur yang relevan dan menghasilkan performansi yang optimal dalam klasifikasi pneumonia.

### 3.4 	Ekstraksi Fitur Menggunakan VGG-16
Ekstraksi fitur merupakan langkah krusial dalam analisis citra digital, khususnya untuk aplikasi klasifikasi seperti dalam penelitian ini. Model VGG-16 digunakan untuk mengekstraksi fitur dari citra input. VGG-16 adalah model deep learning berbasis Convolutional Neural Networks (CNN) yang dikenal karena arsitekturnya yang sederhana namun efektif. Model ini dirancang untuk mempelajari fitur hierarkis dari citra melalui serangkaian lapisan konvolusional yang diikuti oleh pooling dan fully connected layer.

#### 3.4.1. Arsitektur VGG-16 untuk Ekstraksi Fitur
VGG-16 memiliki 13 lapisan konvolusional dan 3 lapisan fully connected, dengan total 16 lapisan utama yang dapat dilatih. Dalam proses ekstraksi fitur, hanya lapisan konvolusional yang digunakan karena tujuan utamanya adalah untuk menghasilkan representasi fitur dari citra input. Lapisan fully connected tidak digunakan pada tahap ekstraksi fitur, karena fungsinya hanya diperlukan pada klasifikasi akhir.

Citra input diubah ukurannya menjadi 224x224 piksel dengan 3 channel warna RGB, sesuai dengan kebutuhan model. Melalui lapisan konvolusional, VGG-16 mengekstraksi fitur spasial dan tekstur dengan resolusi tinggi pada level awal, dan abstraksi yang lebih kompleks pada level yang lebih dalam.

#### 3.4.2. Ekstraksi Fitur Hierarkis
Lapisan-lapisan VGG-16 dirancang untuk mempelajari fitur pada berbagai level abstraksi:
•	Lapisan awal (Conv1 dan Conv2): Menangkap fitur dasar seperti tepi, tekstur, dan pola kecil.
•	Lapisan menengah (Conv3 dan Conv4): Mengidentifikasi bentuk, sudut, atau komponen objek yang lebih kompleks.
•	Lapisan dalam (Conv5): Menangkap fitur abstrak, seperti pola global yang menunjukkan struktur atau kategori dari citra medis.
Setelah melewati lapisan konvolusional terakhir, fitur-fitur yang dihasilkan memiliki dimensi tertentu yang merepresentasikan citra dalam bentuk vektor numerik. Representasi ini digunakan sebagai masukan untuk model klasifikasi.

#### 3.4.3. Transfer Learning untuk Ekstraksi Fitur
VGG-16 menggunakan konsep transfer learning, di mana bobot model yang telah dilatih pada dataset besar seperti ImageNet digunakan untuk ekstraksi fitur. Transfer learning memungkinkan VGG-16 mengenali pola umum yang relevan dengan banyak tugas analisis citra, termasuk klasifikasi pneumonia. Dalam penelitian ini, model pretrained VGG-16 dimodifikasi dengan menghilangkan lapisan fully connected untuk memfokuskan pada ekstraksi fitur.

#### 3.4.4. Dimensi Fitur yang Diekstraksi
Setelah melewati lapisan konvolusional, citra yang awalnya berdimensi (224x224x3) direduksi menjadi tensor fitur berdimensi lebih kecil. Tensor ini berisi nilai numerik yang mewakili karakteristik spesifik dari citra medis. Dimensi tensor tergantung pada arsitektur VGG-16, di mana lapisan terakhir menghasilkan tensor 7x7x512. Tensor ini kemudian diratakan (flattening) menjadi vektor satu dimensi.

#### 3.4.5. Integrasi dengan Extreme Learning Machine (ELM)
Fitur yang telah diekstraksi menggunakan VGG-16 tidak langsung diklasifikasikan oleh lapisan fully connected bawaan, tetapi digunakan sebagai masukan untuk model Extreme Learning Machine (ELM). Hal ini dilakukan karena ELM menawarkan kecepatan pelatihan yang lebih tinggi dibandingkan lapisan fully connected konvensional. ELM bekerja dengan mempelajari pola dalam fitur yang telah diratakan dari model VGG-16.

#### 3.4.6. Keunggulan Ekstraksi Fitur Menggunakan VGG-16
Ekstraksi fitur menggunakan VGG-16 memberikan beberapa keunggulan:
1.	Efisiensi: Dengan menggunakan model pretrained, waktu pelatihan berkurang drastis.
2.	Akurasi Tinggi: Arsitektur mendalam VGG-16 memastikan bahwa semua detail penting dari citra terekstraksi secara optimal.
3.	Generalisasi yang Baik: Fitur yang dihasilkan bersifat robust terhadap variasi citra.

#### Kesimpulan
Ekstraksi fitur menggunakan VGG-16 adalah langkah esensial dalam penelitian ini untuk menghasilkan representasi fitur yang berkualitas tinggi. Kombinasi antara VGG-16 sebagai ekstraktor fitur dan ELM sebagai pengklasifikasi memungkinkan model untuk mencapai performa optimal dalam klasifikasi pneumonia berdasarkan variasi input citra dan komposisi warna HSV.

### 3.5 	Klasifikasi Menggunakan Extreme Learning Machine (ELM)
Extreme Learning Machine (ELM) adalah metode pembelajaran mesin berbasis feedforward neural network yang dirancang untuk menyelesaikan tugas klasifikasi dan regresi dengan efisiensi tinggi. ELM terkenal karena kecepatan pelatihannya yang luar biasa dibandingkan dengan algoritma pembelajaran lainnya, seperti backpropagation pada neural network konvensional. Pada penelitian ini, ELM digunakan sebagai pengklasifikasi untuk memproses fitur-fitur yang diekstraksi dari model VGG-16.

#### 3.5.1. Struktur dan Mekanisme ELM
ELM adalah model jaringan saraf yang terdiri dari tiga komponen utama:
•	Input Layer: Menerima vektor fitur hasil ekstraksi menggunakan VGG-16.
•	Hidden Layer: Parameter pada layer tersembunyi (bobot dan bias) diinisialisasi secara acak dan tidak dioptimalkan selama pelatihan, sehingga mengurangi waktu komputasi.
•	Output Layer: Melakukan klasifikasi berdasarkan fitur yang telah diproses di layer tersembunyi.
Ciri khas ELM adalah pelatihan hanya melibatkan penghitungan bobot output melalui solusi analitik, menggunakan inversi matriks pseudo. Ini berbeda dengan algoritma seperti backpropagation yang memerlukan iterasi untuk memperbarui bobot melalui propagasi error.

#### 3.5.2. Proses Klasifikasi Menggunakan ELM
Dalam penelitian ini, hasil ekstraksi fitur yang diperoleh dari lapisan konvolusional terakhir pada model VGG-16 digunakan sebagai input ke ELM. Proses klasifikasi menggunakan ELM mencakup langkah-langkah berikut:
1.	Input Data: Vektor fitur hasil ekstraksi dari model VGG-16 dimasukkan ke layer input ELM.
2.	Inisialisasi Hidden Layer: Parameter pada hidden layer diinisialisasi secara acak.
3.	Pelatihan Model: Bobot antara hidden layer dan output layer dihitung secara langsung menggunakan solusi least-squares.
4.	Prediksi Kelas: Model memetakan fitur ke kelas output berdasarkan bobot yang telah dihitung.
Keunggulan metode ini adalah kecepatan komputasi yang memungkinkan pengolahan dataset besar secara efisien.

#### 3.5.3. Fungsi Aktivasi dan Pengaruhnya
Pemilihan fungsi aktivasi pada hidden layer memiliki pengaruh signifikan terhadap kinerja ELM. Beberapa fungsi aktivasi yang umum digunakan meliputi:
•	ReLU (Rectified Linear Unit): Membantu mengatasi masalah vanishing gradient dan memberikan hasil yang robust.
•	Sigmoid: Sering digunakan pada masalah klasifikasi biner.
•	Tanh: Cocok untuk data yang memiliki distribusi non-linear.
Studi oleh Ahmad, et al. (2024) menunjukkan bahwa ELM dengan fungsi aktivasi ReLU menghasilkan akurasi yang lebih tinggi dalam klasifikasi citra medis dibandingkan fungsi aktivasi lainnya.

#### 3.5.4. Integrasi dengan VGG-16
Integrasi antara VGG-16 dan ELM memberikan manfaat tambahan. VGG-16 berfungsi sebagai ekstraktor fitur yang kuat, sedangkan ELM memberikan kecepatan pelatihan yang luar biasa. Kombinasi ini memastikan bahwa sistem dapat menangani kompleksitas data citra medis dengan efisien.

Pada tahap ini, hasil ekstraksi fitur dari lapisan konvolusional terakhir model VGG-16 digunakan sebagai input untuk ELM. Pendekatan ini mengurangi dimensi data dan memastikan bahwa hanya informasi yang paling relevan yang diteruskan ke pengklasifikasi, sehingga meningkatkan akurasi klasifikasi.

#### 3.5.5. Evaluasi Kinerja ELM
ELM dievaluasi berdasarkan beberapa metrik performa, seperti:
•	Akurasi: Tingkat ketepatan prediksi model.
•	Presisi dan Recall: Mengevaluasi kemampuan model dalam mengenali kelas target.
•	Waktu Pelatihan: Dibandingkan dengan model lain, ELM menunjukkan efisiensi waktu yang lebih baik.
Hasil penelitian ini menunjukkan bahwa integrasi ELM dengan VGG-16 mampu memberikan kinerja yang kompetitif dengan model deep learning murni, namun dengan waktu pelatihan yang jauh lebih singkat.

Kesimpulan
Penggunaan ELM sebagai pengklasifikasi dalam penelitian ini memberikan beberapa keunggulan utama, seperti kecepatan pelatihan, efisiensi komputasi, dan kinerja klasifikasi yang kompetitif. Ketika digabungkan dengan VGG-16, pendekatan ini menghasilkan model hibrida yang efisien untuk klasifikasi citra medis, khususnya pneumonia. Pemilihan fungsi aktivasi yang tepat dan parameter model menjadi kunci dalam mengoptimalkan performa ELM.

### 3.6 	Evaluasi Model
Evaluasi model merupakan tahapan penting dalam penelitian untuk menilai efektivitas dan efisiensi model klasifikasi yang digunakan. Dalam konteks penelitian ini, evaluasi dilakukan terhadap kinerja Extreme Learning Machine (ELM) yang diintegrasikan dengan fitur-fitur hasil ekstraksi dari model VGG-16. Fokus utama evaluasi adalah untuk menilai seberapa baik model dapat mengklasifikasikan data berdasarkan metrik performa yang relevan.

#### Metrik Evaluasi Model
Dalam evaluasi model menggunakan ELM, beberapa metrik yang umum digunakan adalah:
1.	Akurasi
Akurasi adalah rasio antara jumlah prediksi benar dengan total jumlah data uji. Metrik ini sering digunakan untuk memberikan gambaran umum tentang performa model.
Studi oleh Zhang & Liu (2022) menunjukkan bahwa akurasi merupakan metrik yang penting dalam mengevaluasi klasifikasi citra medis, khususnya untuk dataset yang seimbang.
2.	Precision
Precision mengukur sejauh mana prediksi positif model benar-benar relevan. Precision tinggi menunjukkan bahwa model mampu menghindari prediksi positif palsu (false positives).
3.	Recall (Sensitivitas)
Recall mengukur sejauh mana model dapat mendeteksi semua sampel positif. Dalam aplikasi medis, recall tinggi penting untuk memastikan bahwa semua kasus penyakit teridentifikasi dengan benar.
4.	F1-Score
F1-Score adalah rata-rata harmonis antara precision dan recall. Metrik ini memberikan keseimbangan antara kemampuan model mendeteksi sampel positif dan menghindari prediksi positif palsu.
5.	Confusion Matrix
Confusion matrix memberikan gambaran rinci tentang distribusi prediksi model terhadap kelas sebenarnya, mencakup True Positives, True Negatives, False Positives, dan False Negatives.

#### Analisis Hasil
Setelah model dievaluasi, hasil dari berbagai metrik dibandingkan untuk mendapatkan wawasan tentang kekuatan dan kelemahan model. Hasil ini mencakup:
•	Analisis kesalahan (misalnya, pada kelas tertentu dengan recall rendah).
•	Dampak parameter model (fungsi aktivasi, jumlah hidden nodes, dll.) terhadap performa.
Studi Jayaraman & Ravi (2023) menyoroti bahwa fungsi aktivasi ReLU pada ELM memberikan hasil terbaik untuk data citra non-linear.

## DAFTAR PUSTAKA

World Health Organization. (2021). Air pollution and health. 

Pramudito, A. P., & Pamungkas, A. (2024). Penerapan Deep Learning dalam Pengolahan Citra Digital. Pemrograman Matlab. 

Widyakarya, I. F. (2024). Penerapan Teknologi Pengolahan Citra dalam Analisis Data Visual. Jurnal Teknologi dan Sistem Informasi Widyakarya, 1(1). 

Jain, A., et al. (2021). "Application of Random Forest for Lung Cancer Detection Using X-ray Images." Journal of Healthcare Engineering, 2021, 1-12. 

Chen, X., et al. (2022). "Improved Feature Learning for Speech Recognition Using Deep Neural Networks." IEEE Transactions on Audio, Speech, and Language Processing, 30(2), 476-487. 

Gupta, S., et al. (2023). "Application of Random Forest in Medical Image Classification for Lung Cancer Detection." Journal of Healthcare Engineering, 35(2), 245-258. 

Chen, Y., et al. (2023). "Improved Deep Learning Methods with Transfer Learning for Medical Image Classification." Journal of Medical Systems, 47(1), 1-12. 

Jain, A., & Gupta, M. (2020). Digital Image Processing Techniques in Medical Diagnosis. Journal of Healthcare Engineering, 2020. 

Antony, F., Irsyad, H., & Rivan, M. E. A. (2021). KNN dan Gabor Filter serta Wiener Filter untuk Mendiagnosis Penyakit Pneumonia Citra X-RAY pada Paru-Paru. Jurnal Algoritme, 1(2), 147-155.

Nur A., S. (2023). Dampak pencemaran udara terhadap kesehatan masyarakat di perkotaan. Kompasiana.

Visual Geometry Group. (2023). VGGNet.

Ranjan, A., Kumar, C., Gupta, R. K., & Misra, R. (2020). Transfer learning-based approach for pneumonia detection using customized VGG16 deep learning model. Department of Computer Science and Engineering, Indian Institute of Technology Patna.

Nugraha, S. N., Pebrianto, R., & Fitri, E. (2023). Penerapan deep learning pada klasifikasi tanaman paprika berdasarkan citra daun menggunakan metode CNN. Information System for Educators and Professionals, 8(2), 15-24.

Winangun, P. P., Widyantara, I. M. O., & Hartati, R. S. (2020). Pendekatan diagnostik berbasis extreme learning machine dengan kernel linear untuk mengklasifikasi kelainan paru-paru. Majalah Ilmiah Teknologi Elektro, 19(1), 83-91. 

Nugroho, B., & Puspaningrum, E. Y. (2024). Kinerja metode CNN untuk klasifikasi pneumonia dengan variasi ukuran citra input. Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK).

Sutrisno, S., & Supianto, A. A. (2015). Klasifikasi citra satelit menggunakan kombinasi fitur warna dan fitur tekstur. Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK), 2(2), 102-109.

Ramachandran, R., & Arulselvan, P. (2023). Enhanced Medical Image Classification Using Hybrid CNN-ELM. Biomedical Signal Processing and Control, 82, 104504.

Ahmad, A., & Wani, M. A. (2022). Transfer Learning in Medical Imaging Using VGG Architectures. Journal of Healthcare Informatics Research, 6(1), 89–108.

Guo, Y., & Zhao, Q. (2021). Data Normalization Techniques for Medical Image Analysis. Journal of Computational Medicine, 18(4), 320–331.

Agarwal, R., & Mittal, P. (2020). Preprocessing Techniques for Deep Learning Models in Medical Imaging. International Journal of Biomedical Imaging, 2020.

Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5), 1122–1131.

Cohen, J. P., Morrison, P., & Dao, L. (2020). COVID-19 Image Data Collection: Prospective Predictions Are the Future. arXiv preprint arXiv:2006.11988.

Raja, K., Madan, M., & Kumar, P. (2022). Performance evaluation of hybrid CNN-ELM model for chest X-ray pneumonia classification. Journal of Computational and Applied Mathematics, 389, 113397.

Gupta, A., & Mittal, P. (2021). Pneumonia detection using chest X-ray images and deep learning techniques. Journal of Health Informatics Research, 4(3), 234–249.

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data, 6(1), 60.

