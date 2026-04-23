# Guía de instalación: pyenv + Free-Threaded Python

## 1. Instalar pyenv

```bash
curl https://pyenv.run | bash
```

Agregar al final de `~/.zshrc` (o `~/.bashrc`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reiniciar la terminal o ejecutar `source ~/.zshrc`.

## 2. Instalar Python 3.14.3t (free-threaded)

```bash
# Dependencias de compilación (Manjaro/Arch)
sudo pacman -S base-devel openssl zlib xz tk sqlite libffi bzip2 readline ncurses

# Compilar e instalar (tarda ~2 min)
pyenv install 3.14.3t
```

El sufijo `t` indica la variante free-threaded (compilada con `--disable-gil`).

## 3. Crear el virtualenv

```bash
cd TP1/
uv venv --python ~/.pyenv/versions/3.14.3t/bin/python3.14t
```

## 4. Activar el entorno

```bash
source .venv/bin/activate
```

## 5. Verificar que free-threading está activo

```bash
python -c "import sys; print('GIL enabled:', sys._is_gil_enabled()); print('ABI:', sys.abiflags)"
```

Salida esperada:

```text
GIL enabled: False
ABI: t
```

## Referencia rápida de pyenv

| Comando | Qué hace |
| --- | --- |
| `pyenv install --list` | Listar todas las versiones disponibles |
| `pyenv install 3.14.3t` | Instalar una versión (variantes `t` = free-threaded) |
| `pyenv versions` | Ver versiones instaladas |
| `pyenv uninstall 3.14.3t` | Desinstalar una versión |
| `pyenv shell 3.14.3t` | Usar una versión en la sesión actual |
| `pyenv local 3.14.3t` | Fijar versión para el directorio (crea `.python-version`) |
| `pyenv global 3.14.3t` | Fijar versión global del sistema |
| `pyenv update` | Actualizar pyenv y plugins |

## Dónde vive todo

| Qué | Ruta |
| --- | --- |
| pyenv | `~/.pyenv/` |
| Versiones instaladas | `~/.pyenv/versions/` |
| Binario free-threaded | `~/.pyenv/versions/3.14.3t/bin/python3.14t` |

## Notas sobre free-threading

- El GIL está **desactivado por defecto** en builds `t`. No necesitás hacer nada extra.
- `sys._is_gil_enabled()` devuelve `False` cuando free-threading está activo.
- Extensiones C que no soporten free-threading pueden re-habilitar el GIL automáticamente.
- Para re-habilitar el GIL manualmente: `PYTHON_GIL=1 python script.py`.
- Hay ~5-10% de overhead en single-thread por el reference counting atómico.
