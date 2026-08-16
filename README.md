# Submission 1: Machine Learning Operations (MLOps) - Coronavirus Tweet Sentiment Detection
Nama: Mochamad Yusuf

Username dicoding: mochyusuf

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/) |
| Masalah | Coronavirus Disease 2019 (COVID-19) adalah pandemi yang terjadi sejak tahun 2019 sampai 2023 dimulai dari Wuhan China. Banyak pengguna media sosial terutama twitter yang memberikan tweet mengenai Coronavirus |
| Solusi machine learning |  Solusi machine learning yang akan dibangun adalah sebuah model klasifikasi berbasis teks yang dapat memprediksi apakah tweet yang diberikan mengandung sentimen positif atau tidak dan mengeluarkan prediksi berupa positif, negatif, dan netral  |
| Metode pengolahan | Pada dataset Coronavirus tweets NLP - Text Classification, terdapat 4 fitur, namun dalam proyek ini hanya fitur Original Tweet dan Sentiment yang akan digunakan. Fitur lainnya akan dihapus, selain itu data sudah dipisah menjadi data pelatihan dan evaluasi dengan rasio 80:20. Selain itu, fitur Original Tweet akan diubah menjadi huruf kecil, dan labelnya akan diubah menjadi bentuk integer. |
| Arsitektur model | Berdasarkan hasil parameter tuning, arsitektur model yang digunakan yaitu model embedding dimana terdiri dari vectorize_layer, kemudian layer embedding dengan dimensi embedding yaitu 16, setelah itu layer AveragePooling1D karena data merupakan bentuk text, kemudian layer dense 128, 32, 32 dengan activation relu dan sigmoid karena akan dilakukan klasifikasi antar label. Loss yang digunakan categorical_crossentropy dengan optimizer Adam dan metrik accuracy |
| Metrik evaluasi | Metrik evaluasi yang diterapkan meliputi ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, serta BinaryAccuracy. |
| Performa model | Model hasil pipeline dideploy menggunakan TensorFlow Serving melalui Dockerfile. Pengujian endpoint serving dan prediction request didokumentasikan pada notebook mochyusuf_testing.ipynb. |
