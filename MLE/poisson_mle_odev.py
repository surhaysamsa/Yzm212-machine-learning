"""
Ders: Makine ogrenmesi
Odev: MLE ile Akilli Sehir Planlamasi (Poisson Dagilimi)
Ogrenci: Mustafa Surhay Samsa
Ogrenci Numarasi: 22290159
Tarih: 14.03.2026
"""

import numpy as np
import scipy.optimize as opt
from scipy.stats import poisson
import matplotlib.pyplot as plt


# ============================================================
# GOREV / BOLUM 2: Python ile Sayisal (Numerical) MLE
# ============================================================
# Gozlemlenen Trafik Verisi (1 dakikada gecen arac sayisi)
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])


def negative_log_likelihood(lam, data):
    """
    Poisson dagilimi icin Negatif Log-Likelihood hesaplar.

    Not:
      log(k!) terimi lambda'ya bagli olmadigi icin optimizasyonda sabit kabul edilir.

    Parametreler:
      lam  : scalar veya array benzeri (optimize.minimize uyumlulugu icin)
      data : gozlem dizisi
    """
    lam = np.atleast_1d(lam)[0]
    if lam <= 0:
        return np.inf

    n = len(data)
    sum_k = np.sum(data)

    # log-likelihood (sabit terim disinda): -n*lam + sum(k_i)*log(lam)
    # negatif log-likelihood:
    nll = n * lam - sum_k * np.log(lam)
    return nll


def fit_poisson_mle(data, initial_guess=1.0):
    result = opt.minimize(
        negative_log_likelihood,
        x0=np.array([initial_guess]),
        args=(data,),
        bounds=[(0.001, None)],
    )
    lambda_mle_num = float(result.x[0])
    lambda_mle_analytic = float(np.mean(data))
    return lambda_mle_num, lambda_mle_analytic, result


# Sayisal + Analitik tahminler
lambda_num, lambda_mean, optimization_result = fit_poisson_mle(traffic_data, initial_guess=1.0)

print("=== Bolum 2 Sonuclari ===")
print(f"Sayisal Tahmin (MLE lambda): {lambda_num:.6f}")
print(f"Analitik Tahmin (Ortalama): {lambda_mean:.6f}")
print(f"Fark (Mutlak): {abs(lambda_num - lambda_mean):.6e}")
print(f"Optimizasyon basarili mi?: {optimization_result.success}")
print()


# ============================================================
# GOREV / BOLUM 3: Model Karsilastirma ve Gorsellestirme
# ============================================================
def plot_histogram_and_poisson(data, lam, title_suffix=""):
    k_min = max(0, int(np.min(data)) - 2)
    k_max = int(np.max(data)) + 2
    k_values = np.arange(k_min, k_max + 1)

    pmf_values = poisson.pmf(k_values, mu=lam)

    plt.figure(figsize=(10, 6))

    bins = np.arange(np.min(data) - 0.5, np.max(data) + 1.5, 1)
    plt.hist(data, bins=bins, density=True, alpha=0.55, color="#4C78A8", edgecolor="black", label="Veri Histogrami")

    plt.plot(k_values, pmf_values, marker="o", linestyle="-", color="#F58518", linewidth=2, label=f"Poisson PMF (lambda={lam:.2f})")

    plt.title(f"Poisson Uyum Grafigi {title_suffix}".strip())
    plt.xlabel("1 Dakikadaki Arac Sayisi (k)")
    plt.ylabel("Olasilik / Yogunluk")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


print("=== Bolum 3 ===")
print("Normal veri ile histogram + Poisson PMF grafigi ciziliyor...")
plot_histogram_and_poisson(traffic_data, lambda_num, "(Normal Veri)")


# ============================================================
# GOREV / BOLUM 4: Gercek Hayat Senaryosu - Outlier Analizi
# ============================================================
outlier_data = np.append(traffic_data, 200)

lambda_num_outlier, lambda_mean_outlier, optimization_result_outlier = fit_poisson_mle(
    outlier_data,
    initial_guess=lambda_num,
)

print("=== Bolum 4 Sonuclari (Outlier = 200 Eklendi) ===")
print(f"Yeni Sayisal MLE lambda: {lambda_num_outlier:.6f}")
print(f"Yeni Ortalama: {lambda_mean_outlier:.6f}")
print(f"Lambda artis katsayisi: {lambda_num_outlier / lambda_num:.2f}x")
print()

print("Outlier'li veri ile histogram + Poisson PMF grafigi ciziliyor...")
plot_histogram_and_poisson(outlier_data, lambda_num_outlier, "(Outlier'li Veri)")


# Kisa yorumlar
#Outlier (200) eklendiginde ortalama ciddi bicimde artar. Poisson MLE'de 
#lambda tahmini ortalama oldugu icin model bu tek hatali gozleme cok duyarlidir.
#Bu durum trafik planlamasinda talebi oldugundan yuksek tahmin etmeye,
#dolayisiyla gereksiz kaynak ayirma veya yanlis altyapi kararlarina yol acabilir.
print("=== Kisa Yorum ===")
print(
    "Outlier (200) eklendiginde ortalama ciddi bicimde artar. Poisson MLE'de "
    "lambda tahmini ortalama oldugu icin model bu tek hatali gozleme cok duyarlidir."
)
print(
    "Bu durum trafik planlamasinda talebi oldugundan yuksek tahmin etmeye, "
    "dolayisiyla gereksiz kaynak ayirma veya yanlis altyapi kararlarina yol acabilir."
)
