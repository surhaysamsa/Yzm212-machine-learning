# Çözüm ve Kısa Değerlendirme (Öğrenci Notu)

## 1) Teorik Kısım - Viterbi (high, low)

Verilenler (soruda verilen parametreler):

- Durumlar: `e`, `v`
- Başlangıç: `P(e)=1.0`
- Geçişler:
  - `P(e->e)=0.6`
  - `P(e->v)=0.4`
- Emisyonlar:
  - `P(high|e)=0.7`, `P(low|e)=0.3`
  - `P(high|v)=0.1`, `P(low|v)=0.9`

Gözlem dizisi: `[high, low]`

İlk gözlemde başlangıç yalnızca `e` olduğundan:

`delta_1(e) = P(e)*P(high|e) = 1.0*0.7 = 0.7`

İkinci gözlem (`low`) için iki yol:

1. `e -> e` yolu:

`0.7 * P(e->e) * P(low|e) = 0.7 * 0.6 * 0.3 = 0.126`

2. `e -> v` yolu:

`0.7 * P(e->v) * P(low|v) = 0.7 * 0.4 * 0.9 = 0.252`

Sonuç:

- `P(path=e-e, obs)=0.126`
- `P(path=e-v, obs)=0.252`
- **En olası durum dizisi: `e-v`**

Yani benim anladığım kadarıyla `high, low` gözlemi geldiğinde modelin en mantıklı yolu `e-v` oluyor.

## 2) Uygulama Kısmı Özeti

`src/recognizer.py` dosyasında yaptığım şeyler:

- `EV` ve `OKUL` için ayrı HMM modeli tanımladım.
- Eğitim gözlem dizilerinden `startprob`, `transmat`, `emissionprob` değerlerini çıkardım.
- Yeni bir gözlem dizisini iki modele de verip log-likelihood skorlarını hesapladım.
- Skoru daha yüksek olan modeli tahmin olarak seçtim.

## 3) Analiz Soruları

### Soru 1: Gürültü emisyon olasılıklarını nasıl etkiler?

Anladığım kadarıyla gürültü, gözlem sembollerinin doğru fonem/durum ile eşleşmesini bozuyor.
Bu durumda emisyon dağılımı daha dağınık hale geliyor:

- Gerçekte yüksek olması gereken bir durum, düşük/orta gözlem de üretebilir.
- `P(obs|state)` değerleri daha az ayırt edici hale gelir.
- Modeller arası log-likelihood farkı küçülür, bu da hata ihtimalini artırır.

Pratikte bunu azaltmak için:

- Daha fazla ve çeşitli eğitim verisi toplamak,
- Özellikleri iyileştirmek (MFCC, filtreleme gibi),
- Gürültüye daha dayanıklı modelleme teknikleri kullanmak gerekir.

### Soru 2: Neden günümüzde Viterbi yerine daha karmaşık yapılar (DL) tercih ediliyor?

Benim yorumumla, HMM + Viterbi güçlü bir temel ama gerçek dünyada bazı sınırlamaları var:

- Durum bağımsızlığı ve Markov varsayımı bazı durumlarda fazla kısıtlayıcı kalıyor.
- Karmaşık akustik örüntüleri modelleme kapasitesi sınırlı.
- Büyük veri geldiğinde DL (RNN/LSTM/Transformer) genelde daha iyi sonuç veriyor.

Derin öğrenme modelleri tarafında ise:

- Uçtan uca öğrenme yapılabiliyor,
- Daha zengin temsil öğrenebiliyor,
- Gürültü ve konuşmacı farklılıklarına daha iyi uyum sağlayabiliyor.

Bu yüzden modern konuşma tanıma sistemleri çoğunlukla DL tabanlı ilerliyor.
