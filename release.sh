#!/bin/bash

# Faz com que o script pare imediatamente se der erro em qualquer comando
set -e 

# 1. Lê a versão diretamente do Poetry
VERSION=$(poetry version -s)

# 2. Pega a mensagem de commit passada pelo terminal (ou usa uma padrão)
# Se você não digitar nada, ele usa "chore: release version X.X.X"
COMMIT_MSG=${1:-"chore: release version $VERSION"}

echo "🚀 Iniciando release da versão $VERSION..."

# 3. Deleta a pasta dist atual
echo "🧹 Removendo a pasta dist antiga..."
rm -rf dist/

# 4. Faz o build e publica usando Poetry
echo "📦 Construindo e publicando pacote..."
poetry publish --build

# 5. Executa os comandos Git
echo "🐙 Adicionando ao Git, criando Tag e fazendo Push..."
git add .
git commit -m "$COMMIT_MSG"
git tag "$VERSION"
git push
git push origin "$VERSION"

echo "✅ Sucesso! Versão $VERSION publicada e enviada para o repositório."