# transcri-o-com-slm-local
Aplicação desktop em Python para transcrição local de áudio (Faster-Whisper) com estruturação automática em rascunho de prontuário via SLM local (Ollama), incluindo pipeline de RAG para transcrições longas. 100% local, sem envio de dados a serviços externos.

> ⚠️ **Aviso importante:** esta ferramenta gera apenas **rascunhos de apoio administrativo/editorial**. Ela não realiza diagnóstico, não interpreta achados clínicos e não substitui a avaliação de um profissional responsável. Todo conteúdo gerado deve ser revisado e validado antes de qualquer uso.
---

## Funcionalidades

- Transcrição de áudio em português via **Faster-Whisper**, rodando 100% em CPU
- Gravação de áudio direto do microfone, sem precisar de arquivo externo
- Estruturação automática da transcrição em seções (identificação, motivo, histórico, observações, relato direto, instrumentos, encaminhamentos) usando um **SLM local via Ollama**
- Pipeline de **RAG (Retrieval-Augmented Generation)** para transcrições longas: divide o texto em blocos, gera embeddings e busca os trechos mais relevantes por seção, evitando perda de qualidade em sessões extensas
- Liberação automática de RAM: o modelo de transcrição é descarregado da memória após período de inatividade
- Interface com abas (Transcrição / Documento Estruturado) e exportação de ambos para `.txt`

---

## Dependências

### Bibliotecas Python

Instale com:

```bash
pip install customtkinter faster-whisper sounddevice soundfile numpy
```
As bibliotecas `threading`, `tkinter`, `os`, `json`, `urllib`, `gc` e `re` já vêm inclusas no Python padrão — não precisam de instalação.

### Dependência de sistema (Linux)

O `sounddevice` depende da biblioteca nativa **PortAudio**, que não é instalada via `pip` em distros Linux:

```bash
sudo apt update
sudo apt install libportaudio2 portaudio19-dev
```

No Windows, o PortAudio já vem embutido no pacote do `sounddevice` — não é necessário nenhum passo extra.

### Ollama (obrigatório para a estruturação)

A estruturação do documento depende do [Ollama](https://ollama.com) rodando localmente. Após instalar:

```bash
ollama pull llama3.2:3b   # modelo de geração de texto (estruturação)
ollama pull bge-m3        # modelo de embedding (usado no pipeline de RAG)
```

> O nome do modelo de geração pode ser alterado direto na interface do programa — qualquer modelo já baixado no Ollama pode ser usado.
>
> 
---

## Como executar

```bash
git clone https://github.com/natanramos252/transcri-o-com-slm-local
cd NOME_DO_REPO
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt   # ou instale manualmente, conforme acima

ollama serve                       # caso o Ollama não esteja rodando como serviço
python conversor3.py
```
---

## Arquitetura e decisões de design

- **Tudo local, nada na nuvem**: tanto a transcrição (Faster-Whisper) quanto a estruturação (Ollama) rodam na máquina do usuário, sem chamadas a APIs externas — relevante para contextos com dados sensíveis (ex: LGPD).
- **Cache de modelo com descarregamento automático**: o modelo Whisper é carregado uma vez e reutilizado entre transcrições, sendo liberado da RAM automaticamente após período de inatividade configurável.
- **RAG para transcrições longas**: em vez de enviar o texto inteiro de uma vez (limitado pelo contexto do modelo), transcrições acima de um limite configurável são divididas em blocos com sobreposição, indexadas por embedding e processadas seção por seção, buscando apenas os trechos mais relevantes para cada parte do documento.
- **Prompt restritivo por design**: o modelo é instruído explicitamente a não diagnosticar, não inferir informação ausente e marcar seções sem conteúdo correspondente como "Não relatado na sessão", reduzindo o risco de alucinação.

---

## Limitações conhecidas

- A qualidade da transcrição e da estruturação depende diretamente do tamanho dos modelos escolhidos (Whisper `tiny`/`base` e SLMs pequenos como `llama3.2:3b` são mais leves, porém menos precisos que alternativas maiores).
- Modelos pequenos do Ollama podem ocasionalmente recusar a tarefa por filtros de segurança internos, mesmo com prompt neutro — nesse caso, recomenda-se testar outro modelo.
- O projeto não realiza nenhuma validação clínica do conteúdo gerado; toda saída é um rascunho sujeito a revisão humana obrigatória.

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
