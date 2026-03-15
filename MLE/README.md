# YZM212 Makine Ogrenmesi - 2. Lab Odevi

Bu klasorde, Poisson dagilimi icin Maximum Likelihood Estimation (MLE) odevi cozumlenmistir.

## 1) Problem Tanimi
Sehir trafigindeki 1 dakikada gecen arac sayisi Poisson dagilimina uygun kabul edilmistir.
Veri:

`[12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]`

Amac, Poisson parametresi `lambda` icin MLE tahminini bulmak ve model uyumunu gorsellestirmektir.

## 2) Teorik Cozum (Ozet)
Poisson olasilik kutle fonksiyonu:

`P(k|lambda) = exp(-lambda) * lambda^k / k!`

Bagimsiz gozlemler `k1, ..., kn` icin olabilirlik:

`L(lambda) = Product_i [exp(-lambda) * lambda^(k_i) / k_i!]`

Log-olabilirlik:

`ell(lambda) = -n*lambda + (Sum_i k_i) * log(lambda) - Sum_i log(k_i!)`

Turev sifira esitlenirse:

`d ell / d lambda = -n + (Sum_i k_i)/lambda = 0`

Buradan:

`lambda_hat_MLE = (1/n) * Sum_i k_i = aritmetik ortalama`

## 3) Sayisal Cozum (Python)
`poisson_mle_odev.py` dosyasi su adimlari yapar:
- Negatif log-likelihood fonksiyonunu tanimlar.
- `scipy.optimize.minimize` ile `lambda` degerini sayisal olarak bulur.
- Analitik cozum (`np.mean`) ile karsilastirir.

Calistirma komutu:

```powershell
python poisson_mle_odev.py
```

## 4) Model Karsilastirma ve Gorsellestirme
Kod, iki grafik olusturur:
- Normal veri histogrami + Poisson PMF
- Outlier eklenmis veri histogrami + Poisson PMF

Bu grafikler modelin veriye ne kadar iyi uydugunu gostermektedir.

## 5) Outlier (Aykiri Deger) Analizi
Veriye `200` degeri eklendiginde MLE ciddi artar cunku `lambda_hat = ortalama`dir.
Bu, MLE'nin aykiri degerlere duyarli oldugunu gosterir.

Trafik planlama acisindan olasi risk:
- Talep asiri tahmin edilir.
- Gereksiz yol genisletme/kaynak ayirma kararlarina yol acabilir.

## Dosyalar
- `poisson_mle_odev.py`: tum sayisal cozum ve grafikler
- `README.md`: odev ozeti ve teorik aciklama
- `poisson_mle_odev.ipynb`: ödevin yapıldığı notebook dosyası
- `Mle Rapor.pdf`: ödev raporu
- `Figure_1.png`: normal veri histogramı + Poisson PMF
- `Figure_2.png`: outlier eklenmis veri histogramı + Poisson PMF