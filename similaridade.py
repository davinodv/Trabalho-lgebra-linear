import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset com 20 textos curtos (10 sobre futebol e 10 sobre tecnologia)
docs = [
    # Textos sobre futebol
    "O Liverpool venceu o Manchester City na Premier League.",
    "O Corinthians empatou com o Palmeiras em Itaquera.",
    "O Flamengo conquistou mais uma vitória no Maracanã.",
    "O Santos perdeu fora de casa para o Grêmio.",
    "O goleiro defendeu um pênalti decisivo no final do jogo.",
    "O atacante marcou três gols e foi o destaque da partida.",
    "O time jogou bem, mas desperdiçou muitas chances de gol.",
    "O técnico elogiou o desempenho coletivo dos jogadores.",
    "O zagueiro sofreu uma lesão durante o primeiro tempo.",
    "O campeonato está equilibrado e sem favoritos claros.",

    # Textos sobre tecnologia no esporte
    "A inteligência artificial está mudando a análise de desempenho.",
    "Sensores vestíveis monitoram o esforço físico dos atletas.",
    "O uso de big data ajuda técnicos a tomarem decisões táticas.",
    "Câmeras inteligentes rastreiam a movimentação dos jogadores.",
    "Softwares de análise auxiliam no treinamento personalizado.",
    "A tecnologia VAR revisa lances duvidosos durante as partidas.",
    "Algoritmos de aprendizado de máquina identificam padrões de jogo.",
    "Drones capturam imagens aéreas para análise de posicionamento.",
    "Dispositivos IoT enviam dados em tempo real para os analistas.",
    "Plataformas digitais armazenam estatísticas detalhadas dos atletas."
]

# Vetorização TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Matriz de similaridade de cosseno
similarity_matrix = cosine_similarity(X)

# Cálculo dos ângulos entre os vetores (em graus)
angles = np.degrees(np.arccos(np.clip(similarity_matrix, -1.0, 1.0)))

# Exibição dos resultados
print("\nMatriz de Similaridade (Cosseno):\n", np.round(similarity_matrix, 3))
print("\nMatriz de Ângulos (graus):\n", np.round(angles, 2))

# Identificando os textos mais semelhantes
n = len(docs)
max_sim = -1
pair = (None, None)
for i in range(n):
    for j in range(i + 1, n):
        if similarity_matrix[i, j] > max_sim:
            max_sim = similarity_matrix[i, j]
            pair = (i, j)

print(f"\nOs textos mais semelhantes são {pair[0]} e {pair[1]} com similaridade {max_sim:.3f}")