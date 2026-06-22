import threading
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from faster_whisper import WhisperModel
import sounddevice as sd
import soundfile as sf
import os
import json
import urllib.request
import urllib.error
import gc
import re
import numpy as np

# Configurações globais de aparência do CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class TranscritorApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Configurações da Janela Principal
        self.title("Transcritor & Gravador de Áudio - Faster Whisper")
        self.geometry("700x600")
        self.minsize(600, 500)

        # Variáveis de controle
        self.caminho_arquivo = tk.StringVar()
        self.modelo_selecionado = tk.StringVar(value="tiny")
        self.processando = False
        
        # Variáveis para gravação de áudio
        self.gravando = False
        self.audio_dados = []
        self.samplerate = 16000  # 16kHz é o ideal e padrão para modelos de IA como o Whisper
        self.arquivo_gravado = "audio_capturado.wav"

        # Variáveis para integração com Ollama (estruturação do prontuário)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_modelo = tk.StringVar(value="llama3.2:3b")
        self.estruturando = False

        # Configuração do pipeline de RAG para transcrições longas
        self.ollama_embed_url = "http://localhost:11434/api/embeddings"
        self.ollama_embed_modelo = "bge-m3"          # modelo de embedding local via Ollama
        self.rag_chars_limite = 6000                 # acima disso, ativa o modo RAG por chunks
        self.rag_chunk_tamanho = 1200                # tamanho aproximado de cada chunk (caracteres)
        self.rag_chunk_overlap = 200                 # sobreposição entre chunks (mantém contexto)
        self.rag_top_k = 4                            # quantos chunks recuperar por seção

        # Cache do modelo Whisper carregado (evita recarregar a cada transcrição)
        self.whisper_model = None
        self.whisper_model_nome = None
        self.whisper_unload_timer = None
        self.whisper_unload_minutos = 0  # tempo de inatividade antes de liberar a RAM

        # Inicializa a Interface Gráfica
        self.criar_widgets()

    def criar_widgets(self):
        # --- TÍTULO ---
        self.lbl_titulo = ctk.CTkLabel(
            self,
            text="Transcrição e Captura de Áudio com IA",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.lbl_titulo.pack(pady=(20, 10))

        # --- SELEÇÃO DE MODELO ---
        self.frame_modelo = ctk.CTkFrame(self)
        self.frame_modelo.pack(fill="x", padx=20, pady=5)

        self.lbl_modelo = ctk.CTkLabel(
            self.frame_modelo, text="Tamanho do Modelo:", font=ctk.CTkFont(size=14)
        )
        self.lbl_modelo.pack(side="left", padx=10, pady=10)

        self.combo_modelo = ctk.CTkComboBox(
            self.frame_modelo,
            values=["tiny", "base", "small", "medium"],
            variable=self.modelo_selecionado,
            state="readonly",
        )
        self.combo_modelo.pack(side="left", padx=10, pady=10)

        # --- SELEÇÃO / CAPTURA DE ARQUIVO ---
        self.frame_arquivo = ctk.CTkFrame(self)
        self.frame_arquivo.pack(fill="x", padx=20, pady=5)

        # Botão para buscar arquivo local
        self.btn_procurar = ctk.CTkButton(
            self.frame_arquivo,
            text="Selecionar Arquivo",
            command=self.selecionar_arquivo,
            width=140
        )
        self.btn_procurar.pack(side="left", padx=5, pady=10)

        # NOVO: Botão para Gravar Áudio do Microfone
        self.btn_gravar = ctk.CTkButton(
            self.frame_arquivo,
            text="Gravar Áudio",
            fg_color="#2b7337",       # Cor verde para gravação
            hover_color="#1e5226",
            command=self.alternar_gravacao,
            width=140
        )
        self.btn_gravar.pack(side="left", padx=5, pady=10)

        # Entrada de texto com o caminho
        self.entry_caminho = ctk.CTkEntry(
            self.frame_arquivo,
            textvariable=self.caminho_arquivo,
            placeholder_text="Selecione um arquivo ou grave diretamente do microfone...",
            state="disabled",
        )
        self.entry_caminho.pack(
            side="left", fill="x", expand=True, padx=5, pady=10
        )

        # --- BOTÃO PRINCIPAL / STATUS ---
        self.btn_transcrever = ctk.CTkButton(
            self,
            text="Iniciar Transcrição",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.iniciar_thread_transcricao,
        )
        self.btn_transcrever.pack(pady=15)

        self.lbl_status = ctk.CTkLabel(
            self, text="Aguardando ação...", font=ctk.CTkFont(size=12, slant="italic")
        )
        self.lbl_status.pack(pady=(0, 10))

        # NOVO: Botão para salvar a transcrição em arquivo de texto
        self.btn_salvar = ctk.CTkButton(
            self,
            text="Salvar Transcrição",
            command=self.salvar_transcricao,
            width=180,
        )
        self.btn_salvar.pack(pady=(0, 10))

        # --- ESTRUTURAÇÃO COM IA LOCAL (OLLAMA) ---
        self.frame_ollama = ctk.CTkFrame(self)
        self.frame_ollama.pack(fill="x", padx=20, pady=5)

        self.lbl_ollama = ctk.CTkLabel(
            self.frame_ollama, text="Modelo Ollama:", font=ctk.CTkFont(size=14)
        )
        self.lbl_ollama.pack(side="left", padx=10, pady=10)

        self.entry_ollama_modelo = ctk.CTkEntry(
            self.frame_ollama,
            textvariable=self.ollama_modelo,
            placeholder_text="ex: llama3, mistral, gemma2",
            width=160,
        )
        self.entry_ollama_modelo.pack(side="left", padx=5, pady=10)

        self.btn_estruturar = ctk.CTkButton(
            self.frame_ollama,
            text="Estruturar com IA (Ollama)",
            fg_color="#5b3a9c",
            hover_color="#432b73",
            command=self.iniciar_thread_estruturacao,
            width=220,
        )
        self.btn_estruturar.pack(side="left", padx=5, pady=10)

        # Barra de progresso (indeterminada para dar feedback visual)
        self.progress_bar = ctk.CTkProgressBar(self, mode="indeterminate")

        # --- ÁREA DE TEXTO (ABAS: TRANSCRIÇÃO / DOCUMENTO ESTRUTURADO) ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self.tabview.add("Transcrição")
        self.tabview.add("Documento Estruturado")

        self.txt_resultado = ctk.CTkTextbox(
            self.tabview.tab("Transcrição"), font=ctk.CTkFont(size=13), wrap="word"
        )
        self.txt_resultado.pack(fill="both", expand=True)

        self.txt_estruturado = ctk.CTkTextbox(
            self.tabview.tab("Documento Estruturado"), font=ctk.CTkFont(size=13), wrap="word"
        )
        self.txt_estruturado.pack(fill="both", expand=True)

        # Botão para salvar o documento estruturado
        self.btn_salvar_estruturado = ctk.CTkButton(
            self,
            text="Salvar Documento Estruturado",
            command=self.salvar_estruturado,
            width=220,
        )
        self.btn_salvar_estruturado.pack(pady=(0, 15))

    def selecionar_arquivo(self):
        tipos_arquivos = [
            (
                "Arquivos de Áudio",
                "*.mp3 *.wav *.m4a *.ogg *.flac *.wma *.aac",
            ),
            ("Todos os arquivos", "*.*"),
        ]
        caminho = filedialog.askopenfilename(
            title="Escolha o arquivo de áudio", filetypes=tipos_arquivos
        )
        if caminho:
            self.caminho_arquivo.set(caminho)
            self.atualizar_status("Arquivo pronto para transcrição.")

    def alternar_gravacao(self):
        if not self.gravando:
            # Iniciar gravação
            self.gravando = True
            self.btn_gravar.configure(text="Parar e Salvar", fg_color="#9c2a2a", hover_color="#731f1f")
            self.btn_procurar.configure(state="disabled")
            self.btn_transcrever.configure(state="disabled")
            self.atualizar_status("🔴 Gravando áudio do microfone... Fale agora.")
            
            # Limpa dados anteriores e inicia a thread de captura
            self.audio_dados = []
            threading.Thread(target=self.capturar_audio, daemon=True).start()
        else:
            # Parar gravação
            self.gravando = False
            # O encerramento real e salvamento do arquivo acontecem dentro da thread de captura

    def capturar_audio(self):
        # Callback executada pelo sounddevice a cada bloco de áudio recebido
        def callback(indata, frames, time, status):
            if self.gravando:
                self.audio_dados.append(indata.copy())

        # Abre o fluxo de entrada do microfone (1 canal = Mono)
        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback):
            while self.gravando:
                sd.sleep(100) # Mantém a thread viva enquanto grava
        
        # Processa e salva o arquivo após sair do loop (quando clicar em parar)
        if self.audio_dados:
            import numpy as np
            # Concatena todos os blocos de áudio gravados
            audio_completo = np.concatenate(self.audio_dados, axis=0)
            
            # Salva no disco usando soundfile
            sf.write(self.arquivo_gravado, audio_completo, self.samplerate)
            
            # Atualiza a interface gráfica com segurança (usando o schedule do tkinter)
            self.after(0, self.finalizar_layout_gravacao)

    def finalizar_layout_gravacao(self):
        caminho_absoluto = os.path.abspath(self.arquivo_gravado)
        self.caminho_arquivo.set(caminho_absoluto)
        
        # Restaura os botões ao estado normal
        self.btn_gravar.configure(text="Gravar Áudio", fg_color="#2b7337", hover_color="#1e5226")
        self.btn_procurar.configure(state="normal")
        self.btn_transcrever.configure(state="normal")
        self.atualizar_status("Áudio capturado e selecionado com sucesso!")

    def atualizar_status(self, mensagem):
        self.lbl_status.configure(text=mensagem)

    def salvar_transcricao(self):
        conteudo = self.txt_resultado.get("1.0", tk.END).strip()

        if not conteudo:
            self.atualizar_status("Nada para salvar: a transcrição está vazia.")
            return

        caminho_salvar = filedialog.asksaveasfilename(
            title="Salvar transcrição como...",
            defaultextension=".txt",
            initialfile="transcricao.txt",
            filetypes=[("Arquivo de texto", "*.txt"), ("Todos os arquivos", "*.*")],
        )

        if not caminho_salvar:
            return  # Usuário cancelou a janela

        try:
            with open(caminho_salvar, "w", encoding="utf-8") as f:
                f.write(conteudo)
            self.atualizar_status(f"Transcrição salva em: {caminho_salvar}")
        except Exception as e:
            self.atualizar_status("Erro ao salvar o arquivo.")
            self.txt_resultado.insert(tk.END, f"\n[ERRO AO SALVAR]: {str(e)}")

    def iniciar_thread_estruturacao(self):
        if self.estruturando:
            return

        texto_transcrito = self.txt_resultado.get("1.0", tk.END).strip()
        if not texto_transcrito:
            self.atualizar_status("Erro: não há transcrição para estruturar.")
            return

        self.estruturando = True
        self.btn_estruturar.configure(state="disabled")
        self.txt_estruturado.delete("1.0", tk.END)

        self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar.start()
        self.atualizar_status("Enviando transcrição ao Ollama para estruturação...")

        threading.Thread(
            target=self.processar_estruturacao, args=(texto_transcrito,), daemon=True
        ).start()

    # ---------- Funções auxiliares de RAG (para transcrições longas) ----------

    def gerar_chunks(self, texto):
        """Divide o texto em pedaços com sobreposição, respeitando quebras de frase quando possível."""
        tamanho = self.rag_chunk_tamanho
        overlap = self.rag_chunk_overlap
        chunks = []
        inicio = 0
        n = len(texto)

        while inicio < n:
            fim = min(inicio + tamanho, n)
            pedaco = texto[inicio:fim]

            # Tenta cortar no final de uma frase, não no meio de uma palavra
            if fim < n:
                corte = max(pedaco.rfind(". "), pedaco.rfind("\n"))
                if corte > tamanho * 0.5:  # só corta ali se não perder muito conteúdo
                    fim = inicio + corte + 1
                    pedaco = texto[inicio:fim]

            chunks.append(pedaco.strip())
            inicio = fim - overlap if fim - overlap > inicio else fim

        return [c for c in chunks if c]

    def obter_embedding(self, texto):
        """Chama a API de embeddings do Ollama e retorna um vetor numpy."""
        payload = {"model": self.ollama_embed_modelo, "prompt": texto}
        req = urllib.request.Request(
            self.ollama_embed_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resultado = json.loads(resp.read().decode("utf-8"))
        return np.array(resultado["embedding"], dtype=np.float32)

    def montar_indice_vetorial(self, chunks):
        """Gera embeddings para todos os chunks e retorna como matriz numpy."""
        vetores = [self.obter_embedding(c) for c in chunks]
        return np.vstack(vetores)

    def buscar_chunks_relevantes(self, consulta, chunks, matriz_embeddings, top_k):
        """Retorna os top_k chunks mais similares à consulta (similaridade de cosseno)."""
        vetor_consulta = self.obter_embedding(consulta)

        # Similaridade de cosseno entre a consulta e todos os chunks
        normas = np.linalg.norm(matriz_embeddings, axis=1) * np.linalg.norm(vetor_consulta)
        normas[normas == 0] = 1e-8  # evita divisão por zero
        similaridades = matriz_embeddings @ vetor_consulta / normas

        indices_top = np.argsort(similaridades)[::-1][:top_k]
        indices_top = sorted(indices_top)  # mantém ordem original do texto
        return [chunks[i] for i in indices_top]

    def montar_prompt_secao(self, titulo_secao, instrucao_secao, contexto):
        return (
            "Você é uma ferramenta administrativa de apoio à organização de anotações de "
            "atendimento. Sua função é puramente editorial: extrair e organizar informação já "
            "presente no texto, sem analisar, julgar ou opinar sobre o conteúdo.\n\n"
            "Você NÃO faz avaliação clínica, NÃO sugere hipóteses e NÃO acrescenta informação "
            "que não esteja explicitamente no texto a seguir.\n\n"
            f"Tarefa: com base apenas nos trechos abaixo (extraídos de uma transcrição maior), "
            f"escreva o conteúdo da seção '{titulo_secao}'.\n"
            f"{instrucao_secao}\n"
            "Prefira citar ou parafrasear de forma próxima ao que foi dito, em vez de resumir "
            "de forma vaga. Evite frases genéricas como 'aspectos relevantes foram mencionados'.\n"
            "Se não houver conteúdo correspondente nos trechos, escreva 'Não relatado na sessão'. "
            "Não invente nada que não esteja no texto.\n\n"
            "--- TRECHOS RELEVANTES DA TRANSCRIÇÃO ---\n"
            f"{contexto}\n"
            "--- FIM DOS TRECHOS ---\n"
        )
    
    def chamar_ollama_generate(self, prompt, num_predict=500):
        payload = {
            "model": self.ollama_modelo.get().strip(),
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",   # tempo que o modelo fica na RAM após a resposta
            "options": {
                "num_predict": num_predict,  # limite de tokens gerados (controlado por chamada)
                "num_ctx": 4096,             # tamanho do contexto enviado ao modelo
            },
        }
        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            resultado = json.loads(resp.read().decode("utf-8"))
        return resultado.get("response", "").strip()

    # ---------- Estruturação: rota direta (curta) e rota RAG (longa) ----------

    # Seções do documento: (título, instrução de busca/consulta, instrução de conteúdo)
    SECOES_DOCUMENTO = [
        ("1. Identificação e Contexto",
         "identificação do entrevistado, idade, data da sessão, quem encaminhou, quem acompanha",
         "Inclua nome/identificador, idade, data, quem encaminhou e quem acompanha, se mencionados."),
        ("2. Motivo Relatado",
         "motivo, queixa, razão da sessão, há quanto tempo o problema é percebido, quem notou primeiro",
         "Descreva o motivo relatado, há quanto tempo é percebido e quem notou primeiro."),
        ("3. Histórico Relatado",
         "desenvolvimento, gestação, parto, marcos, histórico escolar, familiar, médico, rotina, sono, telas",
         "Resuma o histórico de desenvolvimento, escolar, familiar, médico e de rotina relatado."),
        ("4. Observações Comportamentais na Sessão",
         "comportamento, postura, atenção, humor, reações observadas durante a sessão",
         "Descreva observações comportamentais relatadas como ocorridas durante a sessão."),
        ("5. Relato Direto do Entrevistado/Responsável",
         "falas, trechos diretos, pontos levantados pelo entrevistado ou responsável",
         "Cite ou parafraseie de forma próxima ao original as falas relevantes."),
        ("6. Instrumentos ou Procedimentos Mencionados",
         "testes, escalas, instrumentos ou procedimentos citados como aplicados ou a aplicar",
         "Liste testes/escalas/procedimentos mencionados, se houver."),
        ("7. Encaminhamentos e Próximos Passos",
         "encaminhamentos, próximos passos, orientações mencionadas",
         "Liste encaminhamentos ou próximos passos mencionados, se houver."),
    ]

    def processar_estruturacao(self, texto_transcrito):
        try:
            if len(texto_transcrito) > self.rag_chars_limite:
                texto_final = self.processar_estruturacao_rag(texto_transcrito)
            else:
                texto_final = self.processar_estruturacao_direta(texto_transcrito)

            aviso = (
                "[RASCUNHO GERADO POR IA LOCAL — NÃO É UM DOCUMENTO CLÍNICO FINAL]\n"
                "Revisar e validar integralmente antes de qualquer uso profissional.\n"
                + "-" * 60 + "\n\n"
            )

            self.txt_estruturado.insert(tk.END, aviso + texto_final)
            self.atualizar_status("Documento estruturado gerado. Revise antes de usar.")
            self.tabview.set("Documento Estruturado")

        except urllib.error.URLError:
            self.atualizar_status("Erro: não foi possível conectar ao Ollama (ele está rodando?).")
            self.txt_estruturado.insert(
                tk.END,
                "[ERRO] Não foi possível conectar ao Ollama em "
                f"{self.ollama_url}.\nVerifique se o serviço está ativo "
                "('ollama serve'), se o modelo de geração foi baixado ('ollama pull "
                f"{self.ollama_modelo.get().strip()}') e, para transcrições longas, "
                f"se o modelo de embedding foi baixado ('ollama pull {self.ollama_embed_modelo}').",
            )
        except Exception as e:
            self.atualizar_status("Erro durante a estruturação do documento.")
            self.txt_estruturado.insert(tk.END, f"[ERRO FATAL]: {str(e)}")

        finally:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.btn_estruturar.configure(state="normal")
            self.estruturando = False

    def processar_estruturacao_direta(self, texto_transcrito):
        """Transcrição curta: manda tudo de uma vez, sem RAG."""
        prompt = (
            "Você é uma ferramenta administrativa de apoio à organização de anotações de "
            "atendimento. Sua função é puramente editorial: extrair e organizar informações "
            "que já estão presentes no texto fornecido, sem analisar, julgar, diagnosticar ou "
            "opinar sobre o conteúdo.\n\n"

            "REGRAS OBRIGATÓRIAS:\n"
            "- Use apenas informações explicitamente ditas na transcrição.\n"
            "- Não infira, não generalize e não complete lacunas com suposições.\n"
            "- Prefira citar ou parafrasear de forma próxima ao que foi dito, em vez de resumir "
            "de forma vaga.\n"
            "- Evite frases genéricas como 'aspectos relevantes foram mencionados' ou 'o "
            "entrevistado relatou diversos pontos' — descreva especificamente o que foi dito.\n"
            "- Se uma seção não tiver conteúdo correspondente na transcrição, escreva "
            "'Não relatado na sessão' — não invente para preencher.\n\n"

            "Organize o conteúdo nas seguintes seções:\n\n"

            "1. IDENTIFICAÇÃO E CONTEXTO\n"
            "Nome ou identificador mencionado, idade, data da sessão (se citada), quem "
            "encaminhou, quem acompanha o entrevistado na sessão.\n\n"

            "2. MOTIVO RELATADO\n"
            "O que foi descrito como razão da busca pelo atendimento, há quanto tempo a "
            "questão é percebida, quem notou primeiro.\n\n"

            "3. HISTÓRICO RELATADO\n"
            "Desenvolvimento (gestação, parto, marcos), histórico escolar (desempenho, "
            "dificuldades, repetências), histórico familiar (composição, dinâmica, casos "
            "similares na família), histórico médico (comorbidades, medicações, outros "
            "profissionais envolvidos), rotina (sono, alimentação, telas, atividades).\n\n"

            "4. OBSERVAÇÕES COMPORTAMENTAIS NA SESSÃO\n"
            "Comportamento, postura, atenção, humor ou reações descritas como observadas "
            "durante a sessão (não da vida cotidiana, apenas o que foi observado na sessão "
            "em si).\n\n"

            "5. RELATO DIRETO DO ENTREVISTADO/RESPONSÁVEL\n"
            "Trechos ou falas específicas mencionadas, próximas do texto original, que "
            "descrevam como a pessoa percebe a própria situação ou a do paciente.\n\n"

            "6. INSTRUMENTOS OU PROCEDIMENTOS MENCIONADOS\n"
            "Testes, escalas ou procedimentos citados como aplicados ou a aplicar.\n\n"

            "7. ENCAMINHAMENTOS E PRÓXIMOS PASSOS\n"
            "Próximos passos, orientações ou encaminhamentos mencionados na sessão.\n\n"

            "Este é um rascunho de organização que será revisado por um profissional "
            "responsável antes de qualquer uso. Mantenha linguagem neutra e formal.\n\n"

            "--- TRANSCRIÇÃO DA SESSÃO ---\n"
            f"{texto_transcrito}\n"
            "--- FIM DA TRANSCRIÇÃO ---\n"
        )
        return self.chamar_ollama_generate(prompt, num_predict=900)  # documento completo: limite maior

    def processar_estruturacao_rag(self, texto_transcrito):
        """Transcrição longa: divide em chunks, indexa por embedding e gera seção por seção."""
        self.after(0, lambda: self.atualizar_status(
            "Transcrição longa detectada — dividindo em blocos e gerando embeddings..."
        ))

        chunks = self.gerar_chunks(texto_transcrito)
        matriz_embeddings = self.montar_indice_vetorial(chunks)

        partes_finais = []
        for titulo, consulta, instrucao in self.SECOES_DOCUMENTO:
            self.after(0, lambda t=titulo: self.atualizar_status(
                f"Gerando seção: {t}..."
            ))

            trechos_relevantes = self.buscar_chunks_relevantes(
                consulta, chunks, matriz_embeddings, self.rag_top_k
            )
            contexto = "\n\n".join(trechos_relevantes)

            prompt_secao = self.montar_prompt_secao(titulo, instrucao, contexto)
            conteudo_secao = self.chamar_ollama_generate(prompt_secao, num_predict=400)  # seção isolada: limite menor

            partes_finais.append(f"{titulo}\n{conteudo_secao}\n")

        return "\n".join(partes_finais)

    def salvar_estruturado(self):
        conteudo = self.txt_estruturado.get("1.0", tk.END).strip()

        if not conteudo:
            self.atualizar_status("Nada para salvar: o documento estruturado está vazio.")
            return

        caminho_salvar = filedialog.asksaveasfilename(
            title="Salvar documento estruturado como...",
            defaultextension=".txt",
            initialfile="documento_estruturado.txt",
            filetypes=[("Arquivo de texto", "*.txt"), ("Todos os arquivos", "*.*")],
        )

        if not caminho_salvar:
            return

        try:
            with open(caminho_salvar, "w", encoding="utf-8") as f:
                f.write(conteudo)
            self.atualizar_status(f"Documento estruturado salvo em: {caminho_salvar}")
        except Exception as e:
            self.atualizar_status("Erro ao salvar o documento estruturado.")
            self.txt_estruturado.insert(tk.END, f"\n[ERRO AO SALVAR]: {str(e)}")

    def iniciar_thread_transcricao(self):
        if self.processando:
            return

        caminho = self.caminho_arquivo.get()
        if not caminho:
            self.atualizar_status("Erro: Por favor, selecione ou grave um áudio primeiro!")
            return

        # Bloqueia a interface
        self.processando = True
        self.btn_transcrever.configure(state="disabled")
        self.btn_procurar.configure(state="disabled")
        self.btn_gravar.configure(state="disabled")
        self.combo_modelo.configure(state="disabled")
        self.txt_resultado.delete("1.0", tk.END)

        # Inicia a animação da barra de progresso
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar.start()

        threading.Thread(
            target=self.processar_transcricao, args=(caminho,), daemon=True
        ).start()

    def processar_transcricao(self, caminho):
        try:
            tam_modelo = self.modelo_selecionado.get()

            # Cancela qualquer descarregamento agendado, já que vamos usar o modelo agora
            self.cancelar_timer_descarregamento()

            # Reaproveita o modelo já carregado se for o mesmo tamanho,
            # evitando recarregar do disco/memória a cada transcrição.
            if self.whisper_model is None or self.whisper_model_nome != tam_modelo:
                self.atualizar_status(f"Carregando modelo '{tam_modelo}' na CPU...")
                self.whisper_model = WhisperModel(
                    tam_modelo,
                    device="cpu",
                    compute_type="int8",   # quantização leve, reduz RAM e uso de CPU
                    cpu_threads=4,         # evita disparar em todos os núcleos de uma vez
                )
                self.whisper_model_nome = tam_modelo
            else:
                self.atualizar_status(f"Reutilizando modelo '{tam_modelo}' já carregado...")

            model = self.whisper_model

            self.atualizar_status("Transcrevendo... Por favor, aguarde.")

            segments, info = model.transcribe(
                caminho,
                beam_size=1,        # greedy decoding: bem mais rápido que beam_size=5
                language="pt",
                vad_filter=True,    # pula trechos de silêncio, reduz processamento real
            )

            texto_identificado = f"Idioma detectado: {info.language} ({info.language_probability*100:.1f}% de certeza)\n"
            texto_identificado += "-" * 50 + "\n\n"

            self.txt_resultado.insert(tk.END, texto_identificado)

            for segment in segments:
                timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
                self.txt_resultado.insert(tk.END, f"{timestamp}{segment.text}\n")
                self.txt_resultado.see(tk.END)

            self.atualizar_status("Transcrição concluída com sucesso!")

        except Exception as e:
            self.atualizar_status(f"Erro durante o processamento.")
            self.txt_resultado.insert(tk.END, f"\n[ERRO FATAL]: {str(e)}")

        finally:
            # Libera a UI de volta ao estado normal
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.btn_transcrever.configure(state="normal")
            self.btn_procurar.configure(state="normal")
            self.btn_gravar.configure(state="normal")
            self.combo_modelo.configure(state="normal")
            self.processando = False

            # Agenda o descarregamento do modelo da RAM após período de inatividade
            self.agendar_timer_descarregamento()

    def cancelar_timer_descarregamento(self):
        if self.whisper_unload_timer is not None:
            self.whisper_unload_timer.cancel()
            self.whisper_unload_timer = None

    def agendar_timer_descarregamento(self):
        self.cancelar_timer_descarregamento()
        self.whisper_unload_timer = threading.Timer(
            self.whisper_unload_minutos * 60, self.descarregar_modelo_whisper
        )
        self.whisper_unload_timer.daemon = True
        self.whisper_unload_timer.start()

    def descarregar_modelo_whisper(self):
        if self.whisper_model is not None:
            self.whisper_model = None
            self.whisper_model_nome = None
            gc.collect()
            self.after(0, lambda: self.atualizar_status(
                f"Modelo Whisper liberado da RAM (inatividade > {self.whisper_unload_minutos} min)."
            ))


if __name__ == "__main__":
    app = TranscritorApp()
    app.mainloop()
