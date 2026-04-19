🏥 Health Insurance Cost Prediction (Sağlık Sigortası Masraf Tahmini)
Bu proje, kişilerin demografik ve fiziksel özelliklerine (yaş, cinsiyet, vücut kitle indeksi, sigara kullanımı vb.) dayanarak tahmini sağlık sigortası masraflarını hesaplayan bir Makine Öğrenmesi (Machine Learning) Regresyon çalışmasıdır.

Projeyi öne çıkaran en önemli özellik; modellerin sadece doğruluk skorlarına bakılmakla kalınmamış, aynı zamanda Eğitim (Train) ve Test metrikleri arasındaki farklar hesaplanarak Overfitting (Aşırı Öğrenme) durumunun özel bir algoritma ile dinamik olarak kontrol edilmiş olmasıdır.

📊 Veri Seti (Dataset)
Kaggle üzerinden alınan insurance.csv veri seti kullanılmıştır. Veri seti şu özellikleri (features) içermektedir:

age: Sigortalının yaşı

sex: Cinsiyet (Kadın / Erkek)

bmi: Vücut Kitle İndeksi (Body Mass Index)

children: Sahip olunan çocuk/bakmakla yükümlü olunan kişi sayısı

smoker: Sigara içme durumu (Evet / Hayır)

region: ABD içindeki yaşanılan bölge

charges: (Hedef Değişken) Sağlık sigortası tarafından fatura edilen bireysel tıbbi masraflar (Dolar)

🚀 Proje Adımları
Keşifçi Veri Analizi (EDA) ve Temizleme: Veri setindeki eksik değerler kontrol edildi, verinin genel dağılımı incelendi.

Veri Ön İşleme (Data Preprocessing):

Kategorik değişkenler (sex, smoker) ikili (binary) formata çevrildi (Map yöntemi ile).

Çoklu kategoriye sahip region değişkenine One-Hot Encoding uygulandı (pd.get_dummies).

Ölçeklendirme (Scaling): Uzaklık temelli algoritmaların (Örn: SVR) sapmasını engellemek için bağımsız değişkenler MinMaxScaler kullanılarak 0-1 aralığına ölçeklendirildi.

Model Kurulumu ve Overfitting Kontrolü: * 8 farklı temel regresyon modeli (Linear, SVR, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM) eğitildi.

Modellerin Train ve Test R2 skorları arasındaki fark hesaplanarak "Aşırı Öğrenme" yapan modeller tespit edildi.

Hiperparametre Optimizasyonu (Hyperparameter Tuning):

Overfitting tuzağına düşen güçlü ağaç modelleri (Decision Tree, Random Forest, XGBoost, LightGBM) ve Gradient Boosting için GridSearchCV ile çapraz doğrulama (CV=5) yapılarak en ideal parametreler bulundu.

🏆 Sonuçlar ve Model Performansı
Optimizasyon işlemleri sonucunda aşırı öğrenme (overfitting) tamamen engellenmiş ve modellerin veriyi genelleme yeteneği maksimize edilmiştir.

En iyi performansı gösteren Şampiyon Modeller:

LightGBM: Test R2: %86.88 | Train R2: %87.93

XGBoost: Test R2: %86.81 | Train R2: %88.31

Gradient Boosting: Test R2: %86.80 | Train R2: %88.66

Model sonuçları, sağlık gibi karmaşık parametrelerin olduğu bir sektörde masraf tahmininin ~%87 doğrulukla yapılabileceğini kanıtlamıştır.

🛠️ Kullanılan Teknolojiler ve Kütüphaneler
Python * Pandas & NumPy: Veri manipülasyonu ve matematiksel işlemler

Scikit-Learn: Veri ön işleme, modelleme, metrik hesaplama ve GridSearchCV

XGBoost & LightGBM: İleri seviye gradient boosting algoritmaları

