from typing import Dict, Tuple, Set, List, Any, Optional


class TuringMachine:
    """Representação de Máquina de Turing com simulador simples.

    Notas rápidas:
    - A fita é representada internamente por um dict[int, str] para posições ilimitadas nas duas direções.
    - A cabeça é um inteiro (posição). O símbolo em posições não escritas é `blank`.
    - Transições: dict[(state, read_symbol)] = (next_state, write_symbol, direction)
      direção é 'L', 'R' ou 'N'.
    """

    def __init__(
        self,
        states: Set[str],
        input_alphabet: Set[str],
        tape_alphabet: Set[str],
        blank: str,
        transitions: Dict[Tuple[str, str], Tuple[str, str, str]],
        start_state: str,
        accept_states: Set[str],
        reject_states: Optional[Set[str]] = None,
    ) -> None:
        self.states = set(states)
        self.input_alphabet = set(input_alphabet)
        self.tape_alphabet = set(tape_alphabet)
        self.blank = blank
        self.transitions = dict(transitions)
        self.start_state = start_state
        self.accept_states = set(accept_states)
        self.reject_states = set(reject_states or set())

        # runtime
        self.tape: Dict[int, str] = {}
        self.head: int = 0
        self.state: str = self.start_state
        self.halted: bool = False

    # ------------------- simulação -------------------
    def reset(self, tape_str: str) -> None:
        """Inicializa a fita a partir de uma string (sequência de símbolos).

        A posição 0 é o primeiro símbolo da string; a cabeça inicia em 0.
        """
        self.tape = {i: s for i, s in enumerate(list(tape_str))}
        self.head = 0
        self.state = self.start_state
        self.halted = False

    def _read(self) -> str:
        return self.tape.get(self.head, self.blank)

    def _write(self, symbol: str) -> None:
        if symbol == self.blank:
            # opcionalmente removemos a posição para manter fita esparsa
            if self.head in self.tape:
                del self.tape[self.head]
        else:
            self.tape[self.head] = symbol

    def step(self) -> Tuple[str, int, str]:
        """Executa um passo da TM.

        Retorna (state, head, read_symbol) após a transição.
        Se não houver transição definida, marca halted e não altera estado.
        """
        if self.halted:
            return (self.state, self.head, self._read())

        cur = self.state
        sym = self._read()
        key = (cur, sym)
        if key not in self.transitions:
            # se não há transição, definimos halted; aceitação se estado em accept_states
            self.halted = True
            return (self.state, self.head, sym)

        next_state, write_sym, direction = self.transitions[key]
        # aplicar
        self._write(write_sym)
        # mover cabeça
        if direction == 'R':
            self.head += 1
        elif direction == 'L':
            self.head -= 1
        # atualizar estado
        self.state = next_state

        #checar aceitação/rejeição
        if self.state in self.accept_states or self.state in self.reject_states:
            self.halted = True

        return (self.state, self.head, self._read())

    def run(self, max_steps: int = 1000) -> Tuple[str, int]:
        """Executa até halt ou max_steps. Retorna (state, steps_executed)."""
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        return (self.state, steps)

    def tape_str(self, window: int = 20) -> Tuple[str, int]:
        """Retorna uma representação de string da fita e o índice relativo da cabeça na string.

        window define quantos símbolos antes/depois incluir (máximo); a função retorna
        uma substring contendo todas as posições não-brancas dentro do intervalo.
        """
        if len(self.tape) == 0:
            s = self.blank
            idx = 0
            return (s, idx)
        min_pos = min(self.tape.keys())
        max_pos = max(self.tape.keys())
        # limitar por window
        left = max(min_pos, self.head - window)
        right = min(max_pos, self.head + window)
        chars = []
        for i in range(left, right + 1):
            chars.append(self.tape.get(i, self.blank))
        head_idx = self.head - left
        return (''.join(chars), head_idx)

    # ------------------- serialização / visualização -------------------
    def to_dict(self) -> Dict[str, Any]:
        trans = {}
        for (s, r), (ns, w, d) in self.transitions.items():
            trans[f"{s},{r}"] = {"next": ns, "write": w, "dir": d}
        return {
            "Q": sorted(self.states),
            "Sigma": sorted(self.input_alphabet),
            "Gamma": sorted(self.tape_alphabet),
            "blank": self.blank,
            "q0": self.start_state,
            "accept": sorted(self.accept_states),
            "reject": sorted(self.reject_states),
            "delta": trans,
        }

    def to_graphviz(self) -> str:
        """Gera DOT agrupando múltiplos rótulos que vão do mesmo par (s->ns).

        Ex: se houver (q,a)->(r,x,R) e (q,b)->(r,y,L) teremos uma única aresta q->r com rótulos 'a→x,R; b→y,L'.
        """
        lines: List[str] = ["digraph turing {", "  rankdir=LR;", "  node [shape = circle];"]
        for q in sorted(self.states):
            shape = "doublecircle" if q in self.accept_states else "circle"
            lines.append(f'  "{q}" [shape={shape}];')

        # agrupar rótulos por aresta
        edge_map: Dict[Tuple[str, str], List[str]] = {}
        for (s, r), (ns, w, d) in self.transitions.items():
            lbl = f"{r}→{w},{d}"
            edge_map.setdefault((s, ns), []).append(lbl)

        for (s, ns), labels in edge_map.items():
            label = '; '.join(labels)
            lines.append(f'  "{s}" -> "{ns}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    # ------------------- serialização completa (spec + runtime snapshot) -------------------
    def spec(self) -> Dict[str, Any]:
        """Retorna a parte estática (especificação) da TM como dict serializável."""
        trans = {}
        for (s, r), (ns, w, d) in self.transitions.items():
            trans_key = f"{s},{r}"
            trans[trans_key] = {"next": ns, "write": w, "dir": d}
        return {
            "Q": sorted(self.states),
            "Sigma": sorted(self.input_alphabet),
            "Gamma": sorted(self.tape_alphabet),
            "blank": self.blank,
            "q0": self.start_state,
            "accept": sorted(self.accept_states),
            "reject": sorted(self.reject_states),
            "delta": trans,
        }

    def to_snapshot(self) -> Dict[str, Any]:
        """Retorna uma snapshot completa (spec + runtime) serializável."""
        spec = self.spec()
        tape_serial = {str(k): v for k, v in self.tape.items()}
        return {
            "spec": spec,
            "tape": tape_serial,
            "head": int(self.head),
            "state": str(self.state),
            "halted": bool(self.halted),
        }

    @staticmethod
    def from_snapshot(snapshot: Dict[str, Any]) -> "TuringMachine":
        """Reconstrói uma TuringMachine a partir de uma snapshot gerada por to_snapshot()."""
        spec = snapshot["spec"]
        # reconstruir transitions
        raw_trans = spec.get("delta", {})
        transitions: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
        for k, v in raw_trans.items():
            if isinstance(k, str) and "," in k:
                s, r = k.split(",", 1)
            else:
                # caso inesperado
                continue
            transitions[(s, r)] = (v["next"], v["write"], v["dir"])

        tm = TuringMachine(set(spec.get("Q", [])), set(spec.get("Sigma", [])), set(spec.get("Gamma", [])), spec.get("blank", "_"), transitions, spec.get("q0"), set(spec.get("accept", [])), set(spec.get("reject", [])))

        # aplicar snapshot runtime
        tape = {int(k): v for k, v in snapshot.get("tape", {}).items()}
        tm.tape = tape
        tm.head = int(snapshot.get("head", 0))
        tm.state = snapshot.get("state", tm.start_state)
        tm.halted = bool(snapshot.get("halted", False))
        return tm

    # ------------------- utilitários -------------------
    @staticmethod
    def sample_majority_tm() -> "TuringMachine":
        """Cria uma TM ilustrativa que simplesmente varre uma fita de 0/1 e aceita no fim.

        Essa TM não calcula maioria, é apenas um exemplo. Use `make_majority_tm_from_length`
        para gerar uma TM que decide maioria assumindo codificação simples (ver utilitário abaixo).
        """
        Q = {"q_start", "q_scan", "q_accept", "q_reject"}
        Sigma = {"0", "1"}
        Gamma = {"0", "1", "_"}
        blank = "_"
        transitions = {
            ("q_start", "0"): ("q_scan", "0", "R"),
            ("q_start", "1"): ("q_scan", "1", "R"),
            ("q_start", "_"): ("q_accept", "_", "N"),
            ("q_scan", "0"): ("q_scan", "0", "R"),
            ("q_scan", "1"): ("q_scan", "1", "R"),
            ("q_scan", "_"): ("q_accept", "_", "N"),
        }
        start = "q_start"
        accept = {"q_accept"}
        reject = {"q_reject"}
        return TuringMachine(Q, Sigma, Gamma, blank, transitions, start, accept, reject)

    @staticmethod
    def make_majority_tm_from_length(n: int) -> "TuringMachine":
        """Gera uma TM simples que decide maioria de 1s em uma fita contendo exatamente n símbolos (0/1).

        Estratégia simples (construção instrutiva):
        - Essa TM é construída apenas para demonstração: ela varre e conta (na lógica do estado)
          — na prática, para n moderado criamos estados q_i_j contando diferença aproximada.
        - Para n grande essa abordagem explode em estados; portanto use apenas para n pequeno (<=15).
        """
        if n <= 0:
            raise ValueError("n deve ser positivo")
        # estados: q_i_d onde i é posição lida (0..n) e d é diferença desloc (offset by +n)
        # para simplicidade: não implementamos contagem completa; em vez disso, criamos uma TM
        # que vai ler e ir para um estado final 'accept' se soma de 1s > n/2 usando estados enumerados
        Q = set()
        Sigma = {"0", "1"}
        Gamma = {"0", "1", "_"}
        blank = "_"
        transitions: Dict[Tuple[str, str], Tuple[str, str, str]] = {}

        # cria estados q_k where k = number of ones seen so far (0..n)
        for k in range(0, n + 1):
            Q.add(f"q_{k}")
        Q.add("q_accept")
        Q.add("q_reject")
        start = "q_0"

        # transições: ao ler 0 ou 1, incrementa contagem de ones e move para a direita
        for k in range(0, n + 1):
            for sym in ("0", "1"):
                if sym == "1":
                    nk = min(n, k + 1)
                else:
                    nk = k
                transitions[(f"q_{k}", sym)] = (f"q_{nk}", sym, "R")

        # ao ler blank após n símbolos, decide por maioria
        for k in range(0, n + 1):
            if k > n // 2:
                transitions[(f"q_{k}", blank)] = ("q_accept", blank, "N")
            else:
                transitions[(f"q_{k}", blank)] = ("q_reject", blank, "N")

        accept = {"q_accept"}
        reject = {"q_reject"}
        return TuringMachine(Q, Sigma, Gamma, blank, transitions, start, accept, reject)

    @staticmethod
    def make_from_automaton_map(autom_map: Dict[str, Any]) -> "TuringMachine":
        """Tenta mapear um automaton.json simples (mapping like 'neutral->happy':1) para uma TM ilustrativa.

        A TM resultante apenas implementa uma leitura simples do rótulo binário:
        - para um vetor de bits na fita, a TM move até o fim e aceita se maioria 1 (construção por comprimento desconhecido
          não é perfeita, mas geramos uma TM construída para o comprimento presente em metadados se fornecidos externamente).
        """
        # construção conservadora: cria uma TM de varredura que aceita se encontrar pelo menos um '1' (exemplo simples)
        Q = {"q_start", "q_scan", "q_accept", "q_reject"}
        Sigma = {"0", "1"}
        Gamma = {"0", "1", "_"}
        blank = "_"
        transitions = {
            ("q_start", "0"): ("q_scan", "0", "R"),
            ("q_start", "1"): ("q_accept", "1", "N"),
            ("q_start", "_"): ("q_reject", "_", "N"),
            ("q_scan", "0"): ("q_scan", "0", "R"),
            ("q_scan", "1"): ("q_accept", "1", "N"),
            ("q_scan", "_"): ("q_reject", "_", "N"),
        }
        return TuringMachine(Q, Sigma, Gamma, blank, transitions, "q_start", {"q_accept"}, {"q_reject"})
