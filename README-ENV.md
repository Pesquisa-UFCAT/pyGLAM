# Criar o ambiente virtual (nome tradicional: venv)
python -m venv venv
py -3.11 -m venv .venv

# Ou com um nome diferente
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt