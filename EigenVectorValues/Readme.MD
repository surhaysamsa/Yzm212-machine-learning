# YZM212 Makine Öğrenmesi — 3. Laboratuvar Değerlendirmesi


Bu rapor, YZM212 Makine Öğrenmesi dersinin üçüncü laboratuvar ödevi için hazırlanmıştır. Çalışma matris manipülasyonu, özdeğer/özvektör teorisi ve manuel implementasyonu kapsamakta; tüm kod uygulamaları `EigenVectorsValues.ipynb` içinde yer almaktadır.

---

## Soru 1 — Makine Öğrenmesinde Matris Manipülasyonu, Özdeğerler ve Özvektörler

### Temel Tanımlar

Matris manipülasyonu ve özdeğer/özvektörler ilk bakışta soyut kavramlar gibi görünse de ML algoritmalarına biraz daha derine inince bu kavramların neredeyse her yerde karşımıza çıktığı görülüyor.

$n$ örnekten ve $m$ özellikten oluşan bir veri seti, $X \in \mathbb{R}^{n \times m}$ boyutunda bir matristir. ML'de temel matris operasyonlarının rolleri:

| Operasyon | ML'deki Karşılığı |
|-----------|-------------------|
| Matris çarpımı $XW$ | Sinir ağında ileri besleme (forward pass) |
| Transpoz $X^T X$ | Kovaryans / Gram matrisi |
| Tersini alma $(X^T X)^{-1}$ | En küçük kareler çözümü |
| SVD / Özayrışım | Boyut indirgeme, PCA |
| Determinant | Hacim ölçümü, Jacobian |

### Özdeğer ve Özvektör

$A \in \mathbb{R}^{n \times n}$ kare matrisi için:

$$A\mathbf{v} = \lambda \mathbf{v}, \quad \mathbf{v} \neq \mathbf{0}$$

$\mathbf{v}$ vektörü **özvektör**, $\lambda$ skaleri ise **özdeğer** olarak adlandırılır. Geometrik yorum: $A$ dönüşümü uygulandığında özvektörün yönü değişmiyor, sadece $\lambda$ kadar ölçekleniyor. Matrisin "doğal eksenleri" olarak düşünülebilir.

Özdeğerler karakteristik denklemden bulunur:

$$\det(A - \lambda I) = 0$$

$n = 2$ için elle çözülebilir; büyük boyutlarda sayısal yöntemler gereklidir — numpy da tam olarak bunu yapmaktadır.

### ML Algoritmalarıyla İlişkisi

**PCA (Principal Component Analysis):**  
Kovaryans matrisi $C = \frac{1}{n} X^T X$ hesaplanıp özayrışımı yapılıyor. Özvektörler maksimum varyans yönlerini (ana bileşenler), özdeğerler ise her yöndeki bilgi miktarını veriyor. En büyük $k$ özdeğere ait özvektörler tutularak boyut indirgeme gerçekleşiyor. Notebook'taki PCA uygulamasında kovaryans matrisi:

$$C = \begin{bmatrix} 2.925 & 2.015 \\ 2.015 & 2.042 \end{bmatrix}$$

Özdeğerler: $\lambda_1 = 4.547$, $\lambda_2 = 0.421$ — manuel implementasyon ve numpy **birebir aynı sonucu** verdi.

**Spektral Kümeleme:**  
Graf Laplasyen $L = D - W$ üzerinden özayrışım yapılıyor. En küçük $k$ özdeğere ait özvektörler K-means için düşük boyutlu uzay oluşturuyor. Doğrusal ayrılamayan kümelerde klasik K-means'ten üstün.

**Google PageRank:**  
Geçiş matrisinin dominant özvektörü (özdeğer = 1'e karşılık gelen) sayfa sıralama skorlarını veriyor. Algoritmada kullanılan Power Iteration, Soru 3'teki implementasyonla doğrudan bağlantılıdır.

**Derin Öğrenme — Ağırlık Başlatma:**  
Ağırlık matrisinin spektral normu (en büyük tekil değer), gradyan patlaması/kayboluşuyla ilişkilidir. Xavier/He başlatması bu bilgi üzerine kurulmuştur.

**SVM Kernel:**  
Kernel matrisi $K$'nın PSD (positive semi-definite) olma koşulu — tüm özdeğerlerin $\geq 0$ olması — geçersiz kernel fonksiyonları tarafından ihlal edilir.

Tüm bu uygulamalardaki ortak tema: **veri içindeki en "önemli" yapıyı/yönü bulmak için özayrışım kullanılıyor.**

### Kaynaklar
- Brownlee, J. *Introduction to Matrices and Matrix Arithmetic for Machine Learning*. https://machinelearningmastery.com/introduction-matrices-machine-learning/
- Brownlee, J. *Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning*. https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/

---

## Soru 2 — `numpy.linalg.eig` Dokümantasyon ve Kaynak Kod İncelemesi

**Referans:** https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html  
**Kaynak Kod:** https://github.com/numpy/numpy/tree/main/numpy/linalg

### Fonksiyon İmzası

```python
numpy.linalg.eig(a) -> EigResult(eigenvalues=w, eigenvectors=v)
```

| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `a` | `(..., M, M) array_like` | Kare matris girişi |
| `w` (çıktı) | `(..., M) ndarray` | Özdeğerler — kompleks olabilir |
| `v` (çıktı) | `(..., M, M) ndarray` | Normalize özvektörler (sütun bazlı) |

> ⚠️ **Kritik:** `v[:, i]` özvektörü `w[i]` özdeğerine karşılık gelir. Satır bazlı erişim (`v[i, :]`) yanlış sonuç verir.

Test matrisi $A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$ için notebook çıktısı:

```
Özdeğerler (w): [5. 2.]
Özvektörler (v):
[[ 0.894427 -0.707107]
 [ 0.447214  0.707107]]

Doğrulama: A @ v[:,0] = λ₀ * v[:,0]  → True
           A @ v[:,1] = λ₁ * v[:,1]  → True
```

### Kaynak Kod Analizi

`numpy/linalg/linalg.py` incelendiğinde `eig`'in aslında iyi yazılmış bir **LAPACK wrapper** olduğu anlaşıldı:

1. **Giriş doğrulama:** `_makearray` + `_assert_stacked_square`
2. **Tip belirleme:** `_commonType` → `float64` veya `complex128`
3. **LAPACK çağrısı:**
   - Reel matrisler: `dgeev` (Double General EigenValue decomposition)
   - Kompleks matrisler: `zgeev`
4. **Sonuç dönüşü:** `EigResult` namedtuple

### `dgeev` Algoritması

Fonksiyon karakteristik polinom **çözmüyor**. Bunun yerine:

1. **Hessenberg indirgeme:** Householder yansımaları ile $A \to H$ — $O(n^3)$
2. **Çift kaydırmalı QR iterasyonu:** Özdeğerler sayısal olarak bulunuyor
3. **Geriye dönük substitüsyon:** Özvektörler elde ediliyor

Bu, Soru 3'teki naif QR implementasyonumla temelde aynı yaklaşım — fark, numpy'ın **double-shift** mekanizması kullanması. Bu farkın sonuçlara etkisi Soru 3'te somut olarak görülmektedir.

---

## Soru 3 — Numpy `eig` Kullanmadan Manuel Özdeğer Hesaplama

**Referans:** [LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors)

`numpy.linalg.eig` kullanmadan iki farklı yöntemle özdeğer/özvektör hesaplaması yapıldı ve sonuçlar numpy ile karşılaştırıldı.

### Yöntem 1 — Güç İterasyonu (Power Iteration) + Hotelling Deflasyonu

Bu yöntem LucasBN'in referans implementasyonunu temel almaktadır.

**Güç İterasyonu:**
1. Rastgele birim vektör $\mathbf{b}_0$ seç
2. $\mathbf{b}_{k+1} = \dfrac{A \mathbf{b}_k}{\|A \mathbf{b}_k\|}$
3. Rayleigh katsayısı: $\lambda_k = \mathbf{b}_k^T A \mathbf{b}_k$
4. $|\lambda_{k+1} - \lambda_k| < 10^{-12}$ olana kadar devam et

**Hotelling Deflasyonu:**

$$A' = A - \lambda_1 \mathbf{v}_1 \mathbf{v}_1^T$$

Dominant özdeğer/özvektör bulunduktan sonra matrisin o bileşeni çıkarılıyor; işlem $n$ kez tekrarlanıyor.

### Yöntem 2 — QR İterasyonu (Gram-Schmidt)

Gram-Schmidt ortogonalizasyonu ile $A = QR$ ayrıştırması yapıldı; ardından:

$$A_{k+1} = R_k Q_k$$

güncellemesi uygulandı. Yeterli iterasyonda $A_k$ Schur formuna yakınsıyor; köşegen elemanlar özdeğerleri veriyor.

### Gerçek Test Sonuçları

**2×2 Test Matrisi** $A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$:

```
[numpy.linalg.eig]       Özdeğerler: [5. 2.]
[Güç İter. + Deflasyon]  Özdeğerler: [5. 2.]  | max hata: 7.97e-13
[QR İterasyonu]          Özdeğerler: [4.99999 2.00001] | max hata: 1.01e-05
```

**4×4 Simetrik Matris** $B$:

```
[numpy.linalg.eig]       [21.909679  4.094668  0.634537 -5.638884]
[Güç İter. + Deflasyon]  [21.909679  4.094668  0.634537 -5.638884] | max hata: 9.25e-13
[QR İterasyonu]          [21.909679  4.094642  0.634537 -5.638859] | max hata: 2.56e-05
```

**PCA Uygulaması** (kovaryans matrisi $C$):

```
Kovaryans C: [[2.925251 2.015074]
              [2.015074 2.042493]]
Manuel özdeğerler: [4.546719 0.421024]
Numpy  özdeğerler: [4.546719 0.421024]
Fark: < 1e-15  → tam eşleşme
```

**Yakınsama:** Güç iterasyonu 30 adımda `6.92e-13` hataya indi (log-lineer düşüş).

### Karşılaştırmalı Değerlendirme

| Yöntem | 2×2 hata | 4×4 hata | Güçlü Yön | Sınırlama |
|--------|----------|----------|-----------|-----------|
| `numpy.linalg.eig` | — | — | Hız, LAPACK optimizasyonu | Kara kutu |
| **Power+Deflation** | **7.97e-13** | **9.25e-13** | Sezgisel, referansla uyumlu | Asimetrik büyük matrislerde sapma |
| QR İter. (shift yok) | 1.01e-05 | 2.56e-05 | Genel amaçlı, numpy'a yakın fikir | Double-shift olmadan yavaş yakınsama |

Power+Deflation simetrik matrislerde numpy'a pratikte eşdeğer doğruluk verdi. QR'nin hatası daha büyük çünkü shift mekanizması eklenmedi — numpy'ın `dgeev`'indeki double-shift bunu çözüyor.

---

## Dosya Yapısı

```
EigenVectorValues/
├── EigenVectorsValues.ipynb   ← Tüm sorular: teori + kod + görselleştirme
├── Readme.MD                  ← Bu rapor
├── requirements.txt           ← Python bağımlılıkları
└── .gitignore
```

---

## Kaynaklar

1. Brownlee, J. *Introduction to Matrices and Matrix Arithmetic for Machine Learning*. https://machinelearningmastery.com/introduction-matrices-machine-learning/
2. Brownlee, J. *Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning*. https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
3. NumPy Developers. *numpy.linalg.eig Documentation*. https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
4. NumPy Source Code. *numpy/linalg*. https://github.com/numpy/numpy/tree/main/numpy/linalg
5. LucasBN. *Eigenvalues-and-Eigenvectors*. https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
