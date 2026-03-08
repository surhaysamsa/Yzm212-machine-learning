# HMM ile Basit Kelime Tanıma (EV - OKUL)

Bu repo, YZM212 Makine Öğrenmesi dersi ödevi için hazırladığım çalışmadır.
Temel amaç, HMM (Hidden Markov Model) kullanarak gelen bir gözlem dizisinin
"EV" mi yoksa "OKUL" mu olduğuna karar vermektir.

## Klasör Yapısı (Bende Olan)

```text
Makine öğrenmesi/
├── data/
├── src/
│   └── recognizer.py
├── report/
│   └── cozum_anahtari.md
├── requirements.txt
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python src/recognizer.py
```

Programın yaptığı işler kısaca:
1. `EV` ve `OKUL` için iki ayrı HMM modeli oluşturur.
2. Verilen eğitim gözlem dizilerinden geçiş (`A`) ve emisyon (`B`) olasılıklarını çıkarır.
3. Yeni gelen bir gözlem dizisini iki modele de verip log-likelihood skorlarını hesaplar.
4. Skoru daha yüksek olan modeli tahmin olarak seçer.
5. Ek olarak teorik kısımdaki Viterbi örneğini (`high, low`) hesaplar.

## Gözlem Kodlaması

- `0 -> low`
- `1 -> mid`
- `2 -> high`

## Not

- Teorik hesaplar ve analiz cevapları: `report/cozum_anahtari.md`
- Python kodu: `src/recognizer.py`

