import os
import sys
import streamlit as st
import json
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_json
from src.turing import TuringMachine
from src.pipeline import FacialStatePipeline
from src.landmark_extractor import LandmarkExtractor

st.set_page_config(page_title='Facial Digraphs Explorer', layout='wide')

st.title('🎭 Facial Recognition - Análise de Expressões')

col1, col2 = st.columns([1, 2])

with col1:
    st.header('⚙️ Configurações')
    digraphs_dir = st.text_input('📁 Diretório dos digraphs', value='test_images/digraphs')
    
    # Verificar timestamps das imagens vs digraphs
    try:
        import os.path
        neutral_time = os.path.getmtime('test_images/neutral.jpg') if os.path.exists('test_images/neutral.jpg') else 0
        sad_time = os.path.getmtime('test_images/sad.jpg') if os.path.exists('test_images/sad.jpg') else 0
        happy_time = os.path.getmtime('test_images/happy.jpg') if os.path.exists('test_images/happy.jpg') else 0
        digraphs_time = os.path.getmtime('test_images/digraphs/automaton.json') if os.path.exists('test_images/digraphs/automaton.json') else 0
        
        # Verificar se alguma imagem foi modificada após a geração dos digraphs
        images_newer = (neutral_time > digraphs_time) or (sad_time > digraphs_time) or (happy_time > digraphs_time)
        
        if images_newer and digraphs_time > 0:
            st.warning('⚠️ **As imagens foram modificadas!** Clique em "🔄 Regenerar" para atualizar a análise.')
    except Exception:
        pass  # Ignorar erros de timestamp
    
    st.markdown('---')
    st.subheader('🎚️ Ajustar Threshold')
    regen_threshold = st.slider(
        'Threshold de sensibilidade', 
        min_value=0.01, 
        max_value=0.5, 
        value=0.2, 
        step=0.01, 
        help='Valores baixos detectam pequenas mudanças; valores altos exigem mudanças maiores. Recomendado: 0.2'
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        regen_button = st.button('🔄 Regenerar', type='primary', use_container_width=True)
    with col_btn2:
        test_button = st.button('🧪 Testar Thresholds', type='secondary', use_container_width=True)
    
    st.markdown('---')
    run_button = st.button('▶️ Executar Análise', use_container_width=True)

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

def analyze_live(neutral_path, target_path, threshold=2.0):
    """Roda o pipeline diretamente nas imagens BGR e retorna o resultado."""
    p = FacialStatePipeline(threshold=threshold)
    nb = cv2.imread(neutral_path)
    tb = cv2.imread(target_path)
    if nb is None or tb is None:
        return {'label': 'reject', 'reason': 'file_not_found'}
    return p.analyze_pair(nb, tb)

def save_landmarks_overlay(image_path, out_path):
    """Extrai landmarks e salva uma imagem com pontos desenhados. Retorna mensagem de erro ou None."""
    try:
        ext = LandmarkExtractor()
    except Exception as e:
        return f'LandmarkExtractor init failed: {e}'
    img = cv2.imread(image_path)
    if img is None:
        return 'image_not_found'
    try:
        lm = ext.from_bgr(img)
    except Exception as e:
        return f' landmark extraction error: {e}'
    if lm is None:
        return 'no_face_detected'
    out = img.copy()
    for (x, y) in lm:
        cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1)
    try:
        cv2.imwrite(out_path, out)
    except Exception as e:
        return f'write_error: {e}'
    return None

# Processar botão de teste de thresholds
if test_button:
    from src.generate_digraphs import main as gen_main
    import numpy as np
    
    st.markdown("---")
    st.subheader("🧪 Teste Automático de Thresholds")
    st.info("Testando vários valores de threshold para encontrar o melhor...")
    
    # Testar thresholds de 0.05 a 0.3 em incrementos de 0.05
    test_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, th in enumerate(test_thresholds):
        status_text.text(f'Testando threshold {th}...')
        try:
            # Gerar digraphs com este threshold
            gen_main('test_images/neutral.jpg', 'test_images/sad.jpg', 'test_images/happy.jpg', 'test_images/digraphs', threshold=th)
            
            # Carregar os metadados gerados
            sm = try_load_json('test_images/digraphs/diff_neutral_sad_meta.json')
            tm = try_load_json('test_images/digraphs/diff_neutral_happy_meta.json')
            
            if sm and tm:
                # Calcular decisões
                def decide(vec):
                    arr = vec
                    if any((v not in (0,1) for v in arr)):
                        return 'reject'
                    ones = sum(1 for v in arr if v==1)
                    zeros = sum(1 for v in arr if v==0)
                    if ones>zeros: return 'happy'
                    if zeros>ones: return 'sad'
                    return 'neutral'
                
                dec_sad = decide(sm['binary'])
                dec_happy = decide(tm['binary'])
                
                # Calcular estatísticas
                sad_ones = sum(sm['binary'])
                happy_ones = sum(tm['binary'])
                
                results.append({
                    'threshold': th,
                    'neutral->sad': dec_sad,
                    'neutral->happy': dec_happy,
                    'sad_ones': sad_ones,
                    'happy_ones': happy_ones,
                    'sad_total': len(sm['binary']),
                    'happy_total': len(tm['binary'])
                })
        except Exception as e:
            st.warning(f'Erro ao testar threshold {th}: {e}')
        
        progress_bar.progress((i + 1) / len(test_thresholds))
    
    status_text.text('✅ Teste concluído!')
    
    # Exibir resultados em tabela
    st.markdown("### 📊 Resultados dos Testes:")
    
    # Criar DataFrame para exibição
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Adicionar coluna de avaliação (assumindo que queremos sad=sad e happy=happy)
    df['✓ Sad Correto'] = df['neutral->sad'] == 'sad'
    df['✓ Happy Correto'] = df['neutral->happy'] == 'happy'
    df['✓ Ambos Corretos'] = df['✓ Sad Correto'] & df['✓ Happy Correto']
    
    # Estilizar DataFrame
    st.dataframe(
        df[['threshold', 'neutral->sad', 'neutral->happy', 'sad_ones', 'happy_ones', '✓ Ambos Corretos']],
        use_container_width=True
    )
    
    # Encontrar melhor threshold
    best_rows = df[df['✓ Ambos Corretos']]
    if not best_rows.empty:
        # Preferir threshold médio se houver múltiplos corretos
        best_threshold = best_rows.iloc[len(best_rows)//2]['threshold']
        st.success(f"🎯 **Melhor threshold recomendado:** `{best_threshold}`")
        st.info(f"👉 Ajuste o slider para `{best_threshold}` e clique em '🔄 Regenerar'")
    else:
        st.warning("⚠️ Nenhum threshold testado gerou resultados corretos para ambas as expressões.")
        st.info("💡 **Sugestões:**")
        st.markdown("- Verifique se as imagens realmente mostram expressões diferentes")
        st.markdown("- Tente ajustar manualmente o threshold entre os valores testados")
        st.markdown("- Considere usar imagens com expressões mais distintas")

# Processar botão de regeneração
if regen_button:
    from src.generate_digraphs import main as gen_main
    with st.spinner(f'🔄 Regenerando digraphs com threshold={regen_threshold}...'):
        try:
            gen_main('test_images/neutral.jpg', 'test_images/sad.jpg', 'test_images/happy.jpg', 'test_images/digraphs', threshold=regen_threshold)
            st.success(f'✅ Digraphs regenerados com threshold={regen_threshold}!')
            st.info('👉 Clique em "▶️ Executar Análise" para ver os resultados.')
        except Exception as e:
            st.error(f'❌ Erro ao regenerar: {e}')

if run_button:
    ddir = digraphs_dir
    if not os.path.isdir(ddir):
        st.error(f'❌ Diretório inválido: {ddir}')
    else:
            # show summary, automaton, images
            summary = try_load_json(os.path.join(ddir, 'summary.json'))
            autom = try_load_json(os.path.join(ddir, 'automaton.json'))
            
            with st.expander('📁 Ver arquivos gerados (summary.json, automaton.json)', expanded=False):
                st.subheader('Summary')
                st.json(summary if summary else {})
                st.subheader('Automaton (valores calculados a partir da análise real)')
                if autom and '_metadata' in autom:
                    st.success('✅ Este automaton.json foi gerado com base na análise real das suas imagens!')
                    meta = autom.get('_metadata', {})
                    st.markdown(f"**neutral → sad:** {autom.get('neutral->sad')} (rótulo: `{meta.get('neutral->sad_label')}`, ones={meta.get('neutral->sad_ones')}/{meta.get('neutral->sad_total')})")
                    st.markdown(f"**neutral → happy:** {autom.get('neutral->happy')} (rótulo: `{meta.get('neutral->happy_label')}`, ones={meta.get('neutral->happy_ones')}/{meta.get('neutral->happy_total')})")
                    st.markdown(f"**Threshold usado:** {meta.get('threshold')}")
                else:
                    st.warning('⚠️ Este automaton.json parece ser uma versão antiga com valores fixos. Regenere os digraphs para obter valores reais.')
                st.json(autom if autom else {})
            # Mostrar formalização do autômato (Q, Σ, δ, q0, F)
            with st.expander('🔍 Ver formalização teórica do autômato (Q, Σ, δ, q0, F)', expanded=False):
                try:
                    if autom:
                        # autom is expected to be a mapping like {'neutral->happy': 1, 'neutral->sad': 0}
                        st.subheader('Formalização do Autômato (Q, Σ, δ, q0, F)')
                        if '_metadata' in autom:
                            st.success('✅ Autômato gerado com valores reais da análise!')
                        else:
                            st.info('⚠️ Esta é uma representação com valores fixos (versão antiga). Regenere para obter valores reais.')
                        # extrair estados e símbolos a partir das chaves
                        Q = set()
                        transitions = {}
                        for k, v in autom.items():
                            if k == '_metadata':
                                continue  # pular metadados
                            if '->' in k:
                                src, dst = k.split('->')
                                Q.add(src); Q.add(dst)
                                # Se temos metadados, usar o rótulo textual; senão usar o valor numérico
                                if '_metadata' in autom:
                                    label_key = k + '_label'
                                    label = autom['_metadata'].get(label_key, v)
                                else:
                                    label = v
                                transitions.setdefault(src.strip(), []).append((dst.strip(), label))
                            else:
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

                        # Exibir formalização da Máquina de Turing baseada na análise real
                        try:
                            # Tentar carregar a Máquina de Turing gerada
                            tm_json = try_load_json(os.path.join(ddir, 'turing_machine.json'))
                            
                            if tm_json and '_metadata' in tm_json:
                                st.subheader('Máquina de Turing (baseada na análise real)')
                                st.success('✅ Esta Máquina de Turing foi gerada com base na análise real das suas imagens!')
                                
                                # Exibir decisões da análise
                                meta = tm_json.get('_metadata', {})
                                logic = meta.get('logic', {})
                                
                                st.markdown(f"**Decisões da análise:**")
                                st.markdown(f"- neutral → sad: `{meta.get('neutral->sad_decision')}`")
                                st.markdown(f"- neutral → happy: `{meta.get('neutral->happy_decision')}`")
                                st.markdown(f"- Threshold usado: `{meta.get('threshold')}`")
                                
                                # Mostrar lógica de funcionamento
                                st.markdown("**Como a TM funciona:**")
                                st.markdown(f"- 📥 **Entrada `0`:** {logic.get('entrada_0', 'N/A')}")
                                st.markdown(f"- 📥 **Entrada `1`:** {logic.get('entrada_1', 'N/A')}")
                                
                                # manter o mesmo padrão de identificação e tamanho do gráfico (texto maior, gráfico menor)
                                tm_col1, tm_col2 = st.columns([1.6, 1])
                                with tm_col1:
                                    st.markdown('**Formalização (TM):**')
                                    # Criar uma cópia sem _metadata para exibição limpa
                                    tm_display = {k: v for k, v in tm_json.items() if k != '_metadata'}
                                    st.json(tm_display)
                                
                                with tm_col2:
                                    st.markdown('**Visualização da Máquina de Turing (Graphviz):**')
                                    try:
                                        # Criar uma TuringMachine a partir do JSON para gerar o gráfico
                                        from src.turing import TuringMachine
                                        
                                        # Reconstruir transições
                                        transitions = {}
                                        for key, val in tm_json.get('delta', {}).items():
                                            if ',' in key:
                                                state, symbol = key.split(',', 1)
                                                transitions[(state, symbol)] = (val['next'], val['write'], val['dir'])
                                        
                                        tm = TuringMachine(
                                            states=set(tm_json.get('Q', [])),
                                            input_alphabet=set(tm_json.get('Sigma', [])),
                                            tape_alphabet=set(tm_json.get('Gamma', [])),
                                            blank=tm_json.get('blank', '_'),
                                            transitions=transitions,
                                            start_state=tm_json.get('q0', 'q_start'),
                                            accept_states=set(tm_json.get('accept', [])),
                                            reject_states=set(tm_json.get('reject', []))
                                        )
                                        st.graphviz_chart(tm.to_graphviz(), use_container_width=True)
                                    except Exception as e:
                                        st.info(f'Não foi possível renderizar o gráfico da TM: {e}')
                            else:
                                # Fallback para TM de exemplo se não houver arquivo gerado
                                tm = TuringMachine.sample_majority_tm()
                                st.subheader('Máquina de Turing (exemplo estático)')
                                st.warning('⚠️ Esta é uma Máquina de Turing de exemplo com valores fixos. Regenere os digraphs para obter uma TM baseada na análise real.')
                                
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
                    st.image(left_path, caption='diff_neutral_sad.png', width='stretch')
                else:
                    st.write('diff_neutral_sad.png não encontrado')
            with col_b:
                if os.path.exists(right_path):
                    st.image(right_path, caption='diff_neutral_happy.png', width='stretch')
                else:
                    st.write('diff_neutral_happy.png não encontrado')

            st.markdown('---')
            st.markdown('## 🎯 RESULTADO DA ANÁLISE (classificação real)')
            st.markdown('Esta seção mostra a **classificação real** baseada na análise dos vetores binários das suas imagens.')
            
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
                    return 'neutral'
                
                dec_sad = decide(sm['binary'])
                dec_happy = decide(tm['binary'])
                
                # Validar se as expressões correspondem ao esperado
                sad_ones = sum(sm['binary'])
                sad_zeros = len(sm['binary']) - sad_ones
                happy_ones = sum(tm['binary'])
                happy_zeros = len(tm['binary']) - happy_ones
                
                # Lógica de validação:
                # neutral→sad deve retornar 'sad' (maioria zeros) para ser válido → retorna 0
                # neutral→happy deve retornar 'happy' (maioria ones) para ser válido → retorna 1
                sad_valid = (dec_sad == 'sad')
                happy_valid = (dec_happy == 'happy')
                
                # Exibir decisão com destaque visual
                st.markdown('---')
                col_dec1, col_dec2 = st.columns(2)
                with col_dec1:
                    if sad_valid:
                        st.success(f"✅ NEUTRAL → SAD")
                        st.metric(label="Resultado", value="0 (sad válido)", help=f"ones={sad_ones}, zeros={sad_zeros}, total={len(sm['binary'])}")
                    else:
                        st.error(f"❌ NEUTRAL → SAD")
                        st.metric(label="Resultado", value=f"✗ (não é sad: {dec_sad})", help=f"ones={sad_ones}, zeros={sad_zeros}, total={len(sm['binary'])}")
                        st.warning(f"A imagem 'sad.jpg' foi classificada como **{dec_sad}**, não como **sad**!")
                
                with col_dec2:
                    if happy_valid:
                        st.success(f"✅ NEUTRAL → HAPPY")
                        st.metric(label="Resultado", value="1 (happy válido)", help=f"ones={happy_ones}, zeros={happy_zeros}, total={len(tm['binary'])}")
                    else:
                        st.error(f"❌ NEUTRAL → HAPPY")
                        st.metric(label="Resultado", value=f"✗ (não é happy: {dec_happy})", help=f"ones={happy_ones}, zeros={happy_zeros}, total={len(tm['binary'])}")
                        st.warning(f"A imagem 'happy.jpg' foi classificada como **{dec_happy}**, não como **happy**!")
                
                # Resumo final
                st.markdown('---')
                if sad_valid and happy_valid:
                    st.success('🎉 **Ambas as expressões foram validadas com sucesso!**')
                    st.markdown('- sad.jpg → **0** (expressão sad confirmada)')
                    st.markdown('- happy.jpg → **1** (expressão happy confirmada)')
                elif sad_valid:
                    st.warning('⚠️ **Apenas a expressão sad foi validada.**')
                    st.markdown('- sad.jpg → **0** ✅ (expressão sad confirmada)')
                    st.markdown(f'- happy.jpg → **✗** ❌ (classificada como {dec_happy}, não happy)')
                elif happy_valid:
                    st.warning('⚠️ **Apenas a expressão happy foi validada.**')
                    st.markdown(f'- sad.jpg → **✗** ❌ (classificada como {dec_sad}, não sad)')
                    st.markdown('- happy.jpg → **1** ✅ (expressão happy confirmada)')
                else:
                    st.error('❌ **Nenhuma expressão foi validada corretamente!**')
                    st.markdown(f'- sad.jpg → **✗** (classificada como {dec_sad}, não sad)')
                    st.markdown(f'- happy.jpg → **✗** (classificada como {dec_happy}, não happy)')
                    st.info('💡 **Dica:** Use o botão "🧪 Testar Thresholds" para encontrar o threshold ideal.')
                
                # Mostrar detalhes técnicos em expander
                with st.expander('🔍 Ver detalhes técnicos'):
                    st.markdown(f"**NEUTRAL → SAD:** ones={sum(sm['binary'])}, zeros={len(sm['binary'])-sum(sm['binary'])}, total={len(sm['binary'])}")
                    st.markdown(f"**NEUTRAL → HAPPY:** ones={sum(tm['binary'])}, zeros={len(tm['binary'])-sum(tm['binary'])}, total={len(tm['binary'])}")
                    st.markdown('**Lógica de decisão:** Se maioria de 1s → happy, se maioria de 0s → sad, se empate → reject')
            else:
                st.write('Meta files ou automaton não encontrados para decisão')

