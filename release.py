import subprocess
import shutil
import sys
import os

def run_command(command, capture=False):
    """Roda um comando no terminal e para o script se der erro."""
    print(f"👉 Executando: {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=capture)
    
    if result.returncode != 0:
        print(f"❌ Erro ao executar o comando: {command}")
        if capture:
            print(result.stderr)
        sys.exit(1)
        
    return result.stdout.strip() if capture else None

def main():
    # 1. Lê a mensagem de commit passada no terminal ou usa uma padrão
    commit_msg = sys.argv[1] if len(sys.argv) > 1 else "fix: update requirements"

    # 2. Pega a versão atual do Poetry
    version = run_command("poetry version -s", capture=True)
    print(f"\n🚀 Iniciando release da versão {version}...\n")

    # 3. Deleta a pasta dist atual (se existir)
    print("🧹 Removendo a pasta dist antiga...")
    if os.path.exists("dist"):
        shutil.rmtree("dist")

    # 4. Faz o build e publica
    print("\n📦 Construindo e publicando pacote...")
    run_command("poetry publish --build")

    # 5. Comandos Git
    print("\n🐙 Sincronizando com o Git...")
    run_command("git add .")
    run_command(f'git commit -m "{commit_msg}"')
    run_command(f'git tag "{version}"')
    run_command("git push")
    run_command(f'git push origin "{version}"')

    print(f"\n✅ Sucesso! Versão {version} publicada e enviada para o repositório.")

if __name__ == "__main__":
    main()