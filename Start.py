import subprocess
import sys
import platform

def verify_pip():
    """
    Verifica se pip è installato. Se non lo è, lo installa usando il metodo corretto
    a seconda del sistema operativo (Windows o macOS/Linux).
    """
    try:
        import pip
        print("Pip è già installato.")
    except ImportError:
        print("Pip non è installato. Procedo con l'installazione...")
        system_name = platform.system()
        if system_name == "Windows":
            # Su Windows usa ensurepip
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            # Aggiorna pip subito dopo
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        elif system_name in ("Darwin", "Linux"):
            # Su macOS/Linux generalmente è già presente, ma se manca:
            # Prova a installare con get-pip.py come fallback
            try:
                subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            except subprocess.CalledProcessError:
                import urllib.request
                import os
                print("Scaricando get-pip.py per installare pip...")
                url = "https://bootstrap.pypa.io/get-pip.py"
                get_pip_script = "get-pip.py"
                urllib.request.urlretrieve(url, get_pip_script)
                subprocess.check_call([sys.executable, get_pip_script])
                os.remove(get_pip_script)
        else:
            raise OSError(f"Sistema operativo non supportato: {system_name}")
        print("Pip installato correttamente.")

def install_packages(package_list):
    """
    Installa una lista di pacchetti usando pip, sempre aggiornando all'ultima versione disponibile.

    Parameters:
        package_list (list): Lista di nomi di pacchetti da installare.
    """
    for package_name in package_list:
        try:
            print(f"Installazione di {package_name} (ultima versione)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", package_name
            ])
            print(f"Pacchetto '{package_name}' installato con successo.")
        except subprocess.CalledProcessError as e:
            print(f"Errore durante l'installazione del pacchetto '{package_name}': {e}")

# Esecuzione
verify_pip()

packages = ["numpy", "matplotlib", "pandas", "scikit-learn", "imbalanced-learn"]
install_packages(packages)

print("Tutti i pacchetti necessari sono stati installati e aggiornati.")
