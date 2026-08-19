# Submission 1: Machine Learning Operations (MLOps) - Coronavirus Tweet Sentiment Detection
Nama: Mochamad Yusuf

Username dicoding: mochyusuf

| | Deskripsi |
| ----------- | ----------- |
| Dataset | Dataset yang digunakan adalah [Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/). Dataset ini berisi 41157 data fitur numerik dan kategorikal yang merepresentasikan Tweet Twitter mengenai Coronavirus untuk mengklasifikasikan 5 sentiment tweet: Extremely Negative, Negative, Neutral, Positive, dan Extremely Positive. |
| Masalah | Coronavirus Disease 2019 (COVID-19) adalah pandemi yang terjadi sejak tahun 2019 sampai 2023 dimulai dari Wuhan China. Banyak pengguna media sosial terutama twitter yang memberikan tweet mengenai Coronavirus |
| Solusi machine learning |  Solusi machine learning yang akan dibangun adalah sebuah model klasifikasi berbasis teks yang dapat memprediksi apakah tweet yang diberikan mengandung sentimen positif atau tidak dan mengeluarkan prediksi berupa Extremely Negative, Negative, Neutral, Positive, dan Extremely Positive |
| Metode pengolahan | Pada dataset Coronavirus tweets NLP - Text Classification, terdapat 4 fitur, namun dalam proyek ini hanya fitur Original Tweet dan Sentiment yang akan digunakan. Fitur lainnya akan dihapus, selain itu data sudah dipisah menjadi data pelatihan dan evaluasi dengan rasio 80:20. Selain itu, fitur Original Tweet akan diubah menjadi huruf kecil, dan Sentiment akan diubah menjadi representasi one-hot agar sesuai untuk klasifikasi multikelas berbentuk vector lima kelas. |
| Arsitektur model | Berdasarkan hasil parameter tuning, arsitektur model yang digunakan yaitu model embedding dimana terdiri dari vectorize_layer, kemudian layer embedding dengan dimensi embedding yaitu 16, setelah itu layer AveragePooling1D karena data merupakan bentuk text, kemudian layer dense unit_1 128, unit_2 64, unit_3 16 dengan activation relu dan softmax karena akan dilakukan klasifikasi antar label. Loss yang digunakan categorical_crossentropy dengan optimizer Adam dan metrik accuracy |
| Metrik evaluasi | Metrik utama adalah CategoricalAccuracy dengan ambang minimal 0.5 agar model dapat dinyatakan blessed. Selain itu, pipeline juga menghitung AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan ExampleCount. |
| Performa model | Model hasil pipeline dideploy menggunakan TensorFlow Serving melalui Dockerfile. Pengujian endpoint serving dan prediction request didokumentasikan pada notebook mochyusuf_testing.ipynb. Evaluasi model diperoleh yaitu AUC sebesar 0.91, kemudian ExampleCount 8272, dengan Categorical Accuracy  0.699. Untuk FalsePositives 2312, TruePositives 5639, TrueNegatives 30776 dan FalseNegatives 2633.|
