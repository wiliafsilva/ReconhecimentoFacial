import os
import sys
import streamlit as st
import json
import cv2
from pathlib import Path

# Garantir que a raiz do projeto esteja em sys.path para que importações
# como `from src.utils import ...` funcionem tanto ao executar a partir
# da raiz quanto ao executar `streamlit run app.py` dentro da pasta `src`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_json
from src.turing import TuringMachine

st.set_page_config(page_title='Facial Digraphs Explorer', layout='wide')

st.title('Facial Digraphs Explorer')

col1, col2 = st.columns([1, 2])

with col1:
    st.header('Inputs')
    # ordem desejada: primeiro gerar a partir de 3 imagens (passo 1), depois usar digraphs existentes (passo 2)
    if 'mode' not in st.session_state:
        st.session_state['mode'] = 'Primeiro passo - Gerar imagens de teste'
    mode = st.radio('Modo', ['Primeiro passo - Gerar imagens de teste', 'Segundo passo - Usar digraphs existentes'], index=0, key='mode')
    if mode == 'Segundo passo - Usar digraphs existentes':
        digraphs_dir = st.text_input('Diretório dos digraphs', value='test_images/digraphs')
    else:
        neutral = st.text_input('Imagem neutral', value='test_images/neutral.jpg')
        sad = st.text_input('Imagem sad', value='test_images/sad.jpg')
        target = st.text_input('Imagem happy', value='test_images/happy.jpg')
        out_dir = st.text_input('Diretório de saída', value='test_images/digraphs')
        threshold = st.number_input('Threshold (normalizado)', value=0.05, step=0.01)

    run_button = st.button('Executar')

with col2:
    st.header('Output')
    output_area = st.empty()

def show_image(path, caption=None, width=400):
    if not os.path.exists(path):
        st.warning(f'Arquivo não encontrado: {path}')
        return
    img = cv2.imread(path)
    if img is None:
        st.warning('Não foi possível ler a imagem: ' + path)
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption=caption, use_container_width=False, width=width)

def try_load_json(path):
    if not os.path.exists(path):
        return None
    try:
        return load_json(path)
    except Exception as e:
        st.error('Erro ao carregar JSON ' + path + ': ' + str(e))
        return None

if run_button:
    if mode == 'Segundo passo - Usar digraphs existentes':
        ddir = digraphs_dir
        if not os.path.isdir(ddir):
            st.error('Diretório inválido: ' + str(ddir))
        else:
            # show summary, automaton, images
            summary = try_load_json(os.path.join(ddir, 'summary.json'))
            autom = try_load_json(os.path.join(ddir, 'automaton.json'))
            output_area.subheader('Summary')
            st.json(summary if summary else {})
            st.subheader('Automaton')
            st.json(autom if autom else {})
            # Mostrar formalização do autômato (Q, Σ, δ, q0, F)
            try:
                if autom:
                    # autom is expected to be a mapping like {'neutral->happy': 1, 'neutral->sad': 0}
                    st.subheader('Formalização do Autômato (Q, Σ, δ, q0, F)')
                    # extrair estados e símbolos a partir das chaves
                    Q = set()
                    transitions = {}
                    for k, v in autom.items():
                        if '->' in k:
                            src, dst = k.split('->')
                            Q.add(src); Q.add(dst)
                            # representar uma transição nomeada pela chave (podemos usar v como rótulo/valor)
                            transitions.setdefault(src.strip(), []).append((dst.strip(), v))
                        else:
                            # entrada não padrão: manter como comentário
                            transitions.setdefault('?', []).append((k, v))

                    Q = sorted(Q)
                    # definir alfabeto simples como conjunto de rótulos únicos das transições
                    Sigma = sorted({str(label) for outs in transitions.values() for (_, label) in outs})
                    q0 = 'neutral' if 'neutral' in Q else (Q[0] if Q else None)
                    # definir F (estados finais) heurística: estados destino que correspondem a rótulos 'happy' ou valores==1
                    F = set()
                    for src, outs in transitions.items():
                        for dst, lab in outs:
                            if isinstance(lab, str) and lab.lower() == 'happy':
                                F.add(dst)
                            if isinstance(lab, (int, float)) and float(lab) == 1.0:
                                F.add(dst)

                    # dividir em colunas: texto (esquerda) e gráfico (direita)
                    # aumentar a coluna de texto e reduzir a coluna do gráfico para diminuir o gráfico
                    form_col, graph_col = st.columns([1.6, 1])
                    with form_col:
                        st.markdown('**Q (estados):** ' + ', '.join(Q))
                        st.markdown('**Σ (alfabeto / rótulos):** ' + ', '.join(Sigma))
                        st.markdown('**q0 (estado inicial):** ' + str(q0))
                        st.markdown('**F (estados finais - heurística):** ' + (', '.join(sorted(F)) if F else 'nenhum detectado'))

                        # mostrar δ como tabela simples
                        st.markdown('**δ (transições):**')
                        for src in sorted(transitions.keys()):
                            for dst, lab in transitions[src]:
                                st.write(f'  {src} -> {dst} [label={lab}]')

                    # gerar gráfico Graphviz para visualização na coluna direita
                    try:
                        # identificar a imagem do autômato
                        with graph_col:
                            st.markdown('**Visualização do Autômato (Graphviz):**')
                            # construir gv e renderizar em seguida
                        gv_lines = ['digraph automaton {', '  rankdir=LR;', '  node [shape = circle];']
                        for s in Q:
                            shape = 'doublecircle' if s in F else 'circle'
                            gv_lines.append(f'  "{s}" [shape={shape}];')
                        for src in sorted(transitions.keys()):
                            for dst, lab in transitions[src]:
                                gv_lines.append(f'  "{src}" -> "{dst}" [label="{lab}"];')
                        gv_lines.append('}')
                        gv = '\n'.join(gv_lines)
                        with graph_col:
                            st.graphviz_chart(gv, use_container_width=True)
                    except Exception:
                        with graph_col:
                            st.info('Não foi possível renderizar o gráfico do autômato.')

                    # Exibir formalização da Máquina de Turing (exemplo) lado a lado
                    try:
                        tm = TuringMachine.sample_majority_tm()
                        st.subheader('Máquina de Turing (exemplo)')
                        # manter o mesmo padrão de identificação e tamanho do gráfico (texto maior, gráfico menor)
                        tm_col1, tm_col2 = st.columns([1.6, 1])
                        with tm_col1:
                            st.markdown('**Formalização (TM):**')
                            st.json(tm.to_dict())
                        with tm_col2:
                            st.markdown('**Visualização da Máquina de Turing (Graphviz):**')
                            try:
                                st.graphviz_chart(tm.to_graphviz(), use_container_width=True)
                            except Exception:
                                st.info('Não foi possível renderizar o gráfico da TM.')
                    except Exception as e:
                        st.warning('Falha ao exibir Máquina de Turing: ' + str(e))
            except Exception as e:
                st.warning('Falha ao exibir formalização do autômato: ' + str(e))

            st.subheader('Imagens')
            img_col1, img_col2, img_col3 = st.columns(3)
            with img_col1:
                show_image(os.path.join('test_images', 'neutral.jpg'), caption='neutral')
            with img_col2:
                show_image(os.path.join('test_images', 'sad.jpg'), caption='sad')
            with img_col3:
                show_image(os.path.join('test_images', 'happy.jpg'), caption='happy')

            st.subheader('Visualizações geradas')
            # mostrar as duas visualizações lado a lado
            col_a, col_b = st.columns(2)
            left_path = os.path.join(ddir, 'diff_neutral_sad.png')
            right_path = os.path.join(ddir, 'diff_neutral_happy.png')
            with col_a:
                if os.path.exists(left_path):
                    st.image(left_path, caption='diff_neutral_sad.png', use_container_width=True)
                else:
                    st.write('diff_neutral_sad.png não encontrado')
            with col_b:
                if os.path.exists(right_path):
                    st.image(right_path, caption='diff_neutral_happy.png', use_container_width=True)
                else:
                    st.write('diff_neutral_happy.png não encontrado')

            st.subheader('Automaton decision')
            # reuse run_automaton logic lightly
            sm = try_load_json(os.path.join(ddir, 'diff_neutral_sad_meta.json'))
            tm = try_load_json(os.path.join(ddir, 'diff_neutral_happy_meta.json'))
            if sm and tm and autom:
                def decide(vec):
                    arr = vec
                    if any((v not in (0,1) for v in arr)):
                        return 'reject'
                    ones = sum(1 for v in arr if v==1)
                    zeros = sum(1 for v in arr if v==0)
                    if ones>zeros: return 'happy'
                    if zeros>ones: return 'sad'
                    return 'reject'
                st.write('NEUTRAL -> SAD :', decide(sm['binary']))
                st.write('NEUTRAL -> HAPPY :', decide(tm['binary']))
            else:
                st.write('Meta files ou automaton não encontrados para decisão')

    else:
        # gerar usando scripts existentes
        from src.generate_digraphs import main as gen_main
        try:
            # garantir variáveis definidas (caso o usuário tenha mudado o modo antes de preencher)
            neutral = locals().get('neutral', 'test_images/neutral.jpg')
            sad = locals().get('sad', 'test_images/sad.jpg')
            happy = locals().get('happy', 'test_images/happy.jpg')
            out_dir = locals().get('out_dir', 'test_images/digraphs')
            threshold = locals().get('threshold', 0.05)
            gen_main(neutral, sad, happy, out_dir, threshold=threshold)
            st.success('Geração concluída. Verifique o diretório: ' + out_dir)
            # após gerar, alternar para o modo de usar digraphs existentes e tentar recarregar
            try:
                st.session_state['mode'] = 'Segundo passo - Usar digraphs existentes'
            except Exception:
                pass
            try:
                rerun = getattr(st, 'experimental_rerun', None)
                if callable(rerun):
                    rerun()
                else:
                    st.info('Recarregue a página manualmente para ver as atualizações.')
            except Exception:
                st.info('Recarregue a página manualmente para ver as atualizações.')
        except Exception as e:
            st.error('Erro na geração: ' + str(e))
