# Reconhecimento Facial (Facial Digraphs)

Descrição detalhada
-------------------
Este projeto implementa um pipeline experimental para extrair "digraphs" (dígrafos) a partir de landmarks faciais, comparar expressões (por exemplo: neutral → happy / sad), gerar visualizações e construir um artefato simples chamado `automaton.json` usado para decisões rápidas.

O foco é prático: extrair padrões locais de mudança de landmarks entre uma imagem neutra e uma imagem com expressão alvo, representar essas mudanças como vetores binários e grafos, e aplicar regras/um autômato simples para classificar emoções faciais.

Principais capacidades
- Extração de landmarks (MediaPipe)
- Construção de grafos de face (cada landmark é um nó)
- Cálculo de grafos de diferença e vetores binários normalizados por escala da face
- Visualizações (PNG) dos pontos/áreas que mais mudaram
- Serialização de artefatos em `test_images/digraphs/` (JSON + imagens)
- UI simples em Streamlit (`src/app.py`) para gerar e inspecionar resultados

Arquivos e responsabilidade de cada módulo
----------------------------------------
Resumo rápido dos módulos em `src/` e o que cada um faz (útil para leitura do código):

- `src/landmark_extractor.py`
	- Extrai landmarks faciais. Prefere MediaPipe (recomendado) e tem um parser para CSVs do OpenFace.
- `src/digraph.py`
	- Funções para construir o digrafo da face (`build_face_digraph`) e gerar o digrafo de diferença entre duas sets de landmarks (`digraph_from_difference`). Também produz o vetor binário de mudança por landmark.
- `src/generate_digraphs.py`
	- Script principal para gerar artefatos a partir de três imagens (neutral, sad, happy): gera grafos, diffs, PNGs de visualização e salva `automaton.json` e `summary.json` no diretório de saída.
- `src/visualize.py`
	- Funções de visualização (matplotlib) para desenhar landmarks e destacar os nós que mais mudaram; gera PNGs utilizados pelo pipeline.
- `src/utils.py`
	- Utilitários: leitura/gravação JSON, conversões de landmarks, cálculo de bbox/escala, mapeamento de índices para regiões (mouth/eyes/brows).
- `src/dfa.py`
	- Uma implementação simples de autômato/heurística (SimpleEmotionDFA) que, a partir do vetor binário e regiões, aplica regras heurísticas para decidir 'happy', 'sad', 'neutral' ou 'reject'.
- `src/run_automaton.py`
	- Script utilitário que carrega `automaton.json` e arquivos de meta (vetores binários) e mostra as decisões (por primeiro símbolo, por maioria etc.).
- `src/pipeline.py`
	- Classe `FacialStatePipeline` que compõe extração → diffs → DFA para fornecer uma análise de par de imagens (neutra+alvo) e retornar rótulos e metadados.
- `src/annotate_diffs.py`
	- Gera imagens anotadas (PNG) com os top-k índices que mais mudaram (útil para inspeção humana).
- `src/inspect_diffs.py`
	- Funções para sumarizar diffs: contagens por região, top changes e estatísticas úteis para debugging.
- `src/turing.py`
	- Implementação de uma classe `TuringMachine` utilitária usada no projeto apenas como demonstração/ilustração. Contém exemplos estáticos (ex.: `sample_majority_tm`) e um conversor simples e conservador `make_from_automaton_map` — **essas TMs são ilustrativas, não derivadas formalmente do pipeline**.
- `src/app.py`
	- UI Streamlit que orquestra geração/inspeção dos digraphs, renderiza grafos/JSONs e contém a navegação para a página de comparação TM vs Autômato (se presente).

Dependências
-------------
As dependências estão em `requirements.txt`. Principais pacotes:

- `mediapipe` (recomendado para extrair landmarks)
- `opencv-python` (leitura/manipulação de imagens)
- `numpy`, `scipy`, `pandas` (manipulação numérica/dados)
- `networkx` (representação de grafos)
- `scikit-learn` (k-NN usado na construção do grafo)
- `matplotlib` (visualizações)
- `streamlit` (UI)

Instalação e execução (Windows - PowerShell)
-----------------------------------------
Siga estes passos para preparar o ambiente e executar o projeto localmente.

1) Criar e ativar um virtualenv (recomendado)

```powershell
python -m venv .venv
# ativar no PowerShell
& .\.venv\Scripts\Activate.ps1
```

2) Instalar dependências

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Observação: `mediapipe` pode requerer wheel binária compatível com sua plataforma; se encontrar erro, verifique a documentação do MediaPipe e instale a versão correspondente ao seu Python/OS.

3) Gerar digraphs a partir de 3 imagens de exemplo

 Coloque três imagens no projeto (ou use as de `test_images`) e rode:

```powershell
python -m src.generate_digraphs --neutral test_images/neutral.jpg --sad test_images/sad.jpg --happy test_images/happy.jpg --out test_images/digraphs
```

Isso gera em `test_images/digraphs/` os arquivos:
- `face_neutral.json`, `face_sad.json`, `face_happy.json` (grafos de face)
- `diff_neutral_sad_graph.json`, `diff_neutral_happy_graph.json` (grafos de diferença)
- `diff_neutral_sad_meta.json`, `diff_neutral_happy_meta.json` (vetores binários e magnitudes)
- `diff_neutral_sad.png`, `diff_neutral_happy.png` (visualizações)
- `automaton.json` e `summary.json`

4) Executar a interface Streamlit (UI principal)

```powershell
streamlit run src/app.py
```

Na barra lateral você encontrará opções para gerar digraphs (passo 1) ou usar um diretório de digraphs já gerado (passo 2). Há também um botão para abrir a página de comparação "TM vs Autômato".

5) Executar o autômato via linha de comando (opcional)

```powershell
python -m src.run_automaton --dir test_images/digraphs
```

Isso imprime decisões baseadas no vetor binário gerado (por símbolo inicial e por maioria) e mostra o `automaton.json` carregado.

6) Executar pipeline direto (API)

Você pode usar `FacialStatePipeline` da biblioteca diretamente em scripts Python:

```python
from src.pipeline import FacialStatePipeline
pipeline = FacialStatePipeline()
res = pipeline.analyze_images('test_images/neutral.jpg', 'test_images/happy.jpg')
print(res['label'])
```

Testes
------
Rodar testes unitários (se existir pytest configurado):

```powershell
pytest -q
```

Observações sobre formalização (autômato vs TM)
----------------------------------------------
- O arquivo `automaton.json` gerado pelo pipeline é um mapeamento simples (ex.: `{"neutral->happy": 1, "neutral->sad": 0}`) que contém decisões/rotulações utilizadas pelo código.
- A apresentação de uma "formalização (Q, Σ, δ, q0, F)" na UI é inferida heurísticamente a partir desses pares chave→valor. Ou seja, o conteúdo bruto do `automaton.json` é real, mas campos como Q, Σ, q0 e F são derivados por heurística e devem ser considerados interpretações, não uma especificação formal produzida pelo pipeline.
- As máquinas de Turing em `src/turing.py` são exemplos ilustrativos e utilitários; elas não são extrações formais do autômato do pipeline. Se precisar de uma tradução formal AF→TM, o projeto pode ser estendido para gerar essa tradução de forma determinística.



