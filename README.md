# Reconhecimento Facial

Sistema de reconhecimento facial com geração de grafos e análise de expressões.

## 🚀 Como executar o projeto

### 1. Clone o repositório
```bash
git clone https://github.com/wiliafsilva/ReconhecimentoFacial.git
cd ReconhecimentoFacial
```

### 2. Crie o ambiente virtual
```bash
python -m venv .venv
```

### 3. Ative o ambiente virtual
**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Instale as dependências
```bash
pip install -r requirements.txt
```

### 5. Execute a aplicação
```bash
streamlit run src/app.py
```

## 📁 Estrutura do projeto
```
├── src/                    # Código fonte
├── test_images/           # Imagens de teste
├── tests/                 # Testes automatizados
├── requirements.txt       # Dependências Python
└── README.md             # Este arquivo
```

## � Imagens de Teste

O projeto inclui uma pasta `test_images/` com exemplos e estrutura para suas imagens:

```bash
# Para testar rapidamente, coloque suas imagens em test_images/:
test_images/
├── neutral.jpg    # Imagem com expressão neutra
├── happy.jpg      # Imagem com expressão alegre  
└── sad.jpg        # Imagem com expressão triste (opcional)
```

### Exemplo de uso com imagens de teste:
```bash
# Processar imagens da pasta test_images
python src/pipeline.py --neutral test_images/neutral.jpg --happy test_images/happy.jpg --visualize --out visualizations
```

## �🔧 Desenvolvimento

### Executar testes
```bash
pytest tests/
```

### Processar imagens e gerar grafos
```bash
# Gerar dígrafos completos (requer 3 imagens: neutra, triste, alegre)
python src/generate_digraphs.py --neutral caminho/neutra.jpg --sad caminho/triste.jpg --happy caminho/alegre.jpg --out out_digraphs

# Pipeline simplificado (requer 2 imagens: neutra e alvo)
python src/pipeline.py --neutral caminho/neutra.jpg --happy caminho/alegre.jpg --threshold 0.05 --visualize --out visualizations

# Analisar diferenças entre expressões
python src/inspect_diffs.py --dir out_digraphs --neutral caminho/neutra.jpg

# Executar autômato nos grafos gerados
python src/run_automaton.py --dir out_digraphs

# Anotar diferenças nas imagens
python src/annotate_diffs.py --image caminho/imagem.jpg --graph caminho/grafo.json --out imagem_anotada.jpg
```

### Parâmetros importantes
- `--threshold`: Limiar de detecção (0.05 = 5% da diagonal da face)
- `--visualize`: Gera visualizações dos grafos
- `--out`: Diretório de saída para resultados