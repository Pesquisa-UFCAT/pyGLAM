# Ambiente de desenvolvimento com uv

Na raiz do projeto, crie ou sincronize o ambiente:

```bash
uv sync --locked
```

O `uv` usa o `pyproject.toml` e as versões registradas no `uv.lock`, cria a
`.venv` e instala o pyGLAM em modo editável: alterações no código ficam disponíveis
sem reinstalar o pacote. O arquivo `.python-version` seleciona Python 3.11 para
desenvolvimento; a compatibilidade declarada do pacote continua sendo Python >=3.10.

Execute todos os testes, tratando avisos inesperados como erros:

```bash
uv run --locked python -W error -m unittest discover -s tests -v
```

Os testes usam `unittest`, incluído no Python, e as dependências do projeto.
Não é necessário ativar a `.venv` nem instalar um executor de testes adicional.

Para abrir o Python com o pacote disponível:

```bash
uv run --locked python
```

Depois de alterar dependências no `pyproject.toml`, atualize o lock e o ambiente:

```bash
uv sync
```

Versione `pyproject.toml`, `.python-version` e `uv.lock`. A pasta `.venv` é local
e já está no `.gitignore`.

## Documentação Sphinx

Instale o grupo de documentação definido no TOML:

```bash
uv sync --locked --group docs
```

Gere o HTML na raiz do projeto, tratando avisos do Sphinx como erros:

```bash
uv run --locked --group docs sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Abra `docs/build/html/index.html` no navegador. Para executar os exemplos das
páginas e das docstrings:

```bash
uv run --locked --group docs sphinx-build -W --keep-going -b doctest docs/source docs/build/doctest
```

As fontes estão em `docs/source`; exemplos dos métodos públicos também ficam
nas docstrings de `pyglam/glam.py` e `pyglam/performance.py`, incluídas por
`autodoc`. A pasta gerada `docs/build/` está no `.gitignore`.

## Publicação automática no GitHub Pages

O workflow [Documentation](.github/workflows/docs.yml) roda em cada push para
`main` e em pull requests destinados a `main`. Ele instala o Python indicado em
`.python-version`, sincroniza o grupo `docs` pelo `uv.lock`, executa os testes da
biblioteca e os exemplos Sphinx e gera o HTML. Avisos ou falhas interrompem a
publicação.

Após um push para `main`, o HTML aprovado é publicado no GitHub Pages. Pull
requests apenas validam. Também é possível executar manualmente em
**Actions → Documentation → Run workflow**; a publicação só ocorre se a branch
selecionada for `main`.

Na configuração do repositório, selecione **Settings → Pages → Build and
deployment → Source → GitHub Actions**. Se o ambiente `github-pages` tiver
restrições de implantação, permita a branch `main`. Essa configuração é feita
uma vez; não é necessário criar um token pessoal ou versionar o HTML gerado.

Inclua `.github/workflows/docs.yml`, `docs/source/`, `.python-version`,
`pyproject.toml`, `uv.lock`, `tests/` e o código atualizado no push. A URL do site
aparece no job **Publish GitHub Pages** após a primeira execução bem-sucedida.

Referência: [workflows personalizados do GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

Referência: [projetos com uv](https://docs.astral.sh/uv/guides/projects/).
