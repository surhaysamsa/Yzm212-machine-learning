import numpy as np
from hmmlearn import hmm


OBS_MAP = {"low": 0, "mid": 1, "high": 2}
INV_OBS_MAP = {v: k for k, v in OBS_MAP.items()}
N_SYMBOLS = len(OBS_MAP)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    # Her satırın toplamını 1 yaparak olasılık dağılımına çevirir.
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def estimate_hmm_params(state_sequences, observation_sequences, n_states, n_symbols, smoothing=1e-3):
    # Başlangıç, geçiş ve emisyon sayımlarını tutacak matrisler.
    startprob = np.zeros(n_states, dtype=float)
    transmat = np.zeros((n_states, n_states), dtype=float)
    emission = np.zeros((n_states, n_symbols), dtype=float)

    for states, obs in zip(state_sequences, observation_sequences):
        # İlk durum başlangıç olasılığına katkı yapar.
        startprob[states[0]] += 1

        for i in range(len(states) - 1):
            # Ardışık durum geçişlerini sayar.
            transmat[states[i], states[i + 1]] += 1

        for s, o in zip(states, obs):
            # Her durumun hangi gözlemi ürettiğini sayar.
            emission[s, o] += 1

    startprob += smoothing
    transmat += smoothing
    emission += smoothing

    startprob /= startprob.sum()
    transmat = normalize_rows(transmat)
    emission = normalize_rows(emission)

    return startprob, transmat, emission


def create_categorical_hmm(n_states: int, startprob, transmat, emission):
    # hmmlearn modeline hesaplanan parametreleri doğrudan atar.
    model = hmm.CategoricalHMM(n_components=n_states, init_params="")
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emission
    return model


def train_word_model(word_name, phonemes, train_sequences):
    # Fonem sayısı kadar gizli durum kullanılır.
    n_states = len(phonemes)

    state_sequences = []
    obs_sequences = []

    for obs in train_sequences:
        states = []
        for idx in range(n_states):
            # Basit yaklaşım: her durum iki kez tekrar edilerek durum dizisi kurulur.
            states.extend([idx, idx])

        target_len = len(obs)
        if len(states) < target_len:
            # Gözlem daha uzunsa son durumu tekrar ederek eşit uzunluk sağlanır.
            states.extend([n_states - 1] * (target_len - len(states)))
        states = states[:target_len]

        state_sequences.append(np.array(states, dtype=int))
        obs_sequences.append(np.array(obs, dtype=int))

    startprob, transmat, emission = estimate_hmm_params(
        state_sequences=state_sequences,
        observation_sequences=obs_sequences,
        n_states=n_states,
        n_symbols=N_SYMBOLS,
    )

    model = create_categorical_hmm(n_states, startprob, transmat, emission)

    return {
        "word": word_name,
        "phonemes": phonemes,
        "model": model,
        "startprob": startprob,
        "transmat": transmat,
        "emission": emission,
    }


def score_sequence(model, obs_sequence):
    # Gözlem dizisini modelin beklediği sütun vektörü formatına çevirir.
    x = np.array(obs_sequence, dtype=int).reshape(-1, 1)
    return model.score(x)


def classify_observation(models, obs_sequence):
    # Her model için log-likelihood hesaplanır.
    scores = {}
    for name, bundle in models.items():
        scores[name] = score_sequence(bundle["model"], obs_sequence)

    predicted = max(scores, key=scores.get)
    return predicted, scores


def viterbi_two_step_example():
    # Ödev kağıdındaki teorik parametreler.
    p_e_to_e = 0.6
    p_e_to_v = 0.4
    p_high_given_e = 0.7
    p_low_given_e = 0.3
    p_low_given_v = 0.9

    # Gözlem dizisi [high, low] için iki olası yolun olasılığı hesaplanır.
    path_ee = 1.0 * p_high_given_e * p_e_to_e * p_low_given_e
    path_ev = 1.0 * p_high_given_e * p_e_to_v * p_low_given_v

    return {
        "P(e->e | high,low)": path_ee,
        "P(e->v | high,low)": path_ev,
        "best_path": "e-v" if path_ev > path_ee else "e-e",
    }


def pretty_print_model(bundle):
    # Model parametrelerini okunur şekilde yazdırır.
    print(f"\n--- {bundle['word']} modeli ---")
    print("Fonem durumları:", bundle["phonemes"])
    print("Start:", np.round(bundle["startprob"], 3))
    print("Geçiş matrisi (A):\n", np.round(bundle["transmat"], 3))
    print("Emisyon matrisi (B):\n", np.round(bundle["emission"], 3))


def main():
    # Gözlem kodlaması: 0=low, 1=mid, 2=high
    train_ev = [
        [2, 2, 1, 0],
        [2, 1, 1, 0],
        [2, 2, 0, 0],
    ]

    train_okul = [
        [1, 1, 2, 1, 0, 0],
        [1, 2, 2, 1, 0, 0],
        [1, 1, 2, 0, 0, 0],
    ]

    model_ev = train_word_model("EV", ["e", "v"], train_ev)
    model_okul = train_word_model("OKUL", ["o", "k", "u", "l"], train_okul)

    models = {"EV": model_ev, "OKUL": model_okul}

    pretty_print_model(model_ev)
    pretty_print_model(model_okul)

    unknown_obs = [2, 1, 0, 0]
    # Bilinmeyen gözlem dizisinin hangi kelimeye daha yakın olduğu bulunur.
    predicted, scores = classify_observation(models, unknown_obs)

    print("\nTest gözlem dizisi:", [INV_OBS_MAP[i] for i in unknown_obs], unknown_obs)
    for word, score in scores.items():
        print(f"{word} log-likelihood: {score:.4f}")
    print(f"Tahmin edilen kelime: {predicted}")

    viterbi_result = viterbi_two_step_example()
    # Teorik kısımdaki iki-adımlı örnek ayrıca gösterilir.
    print("\nViterbi (high, low) örneği:")
    for k, v in viterbi_result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
