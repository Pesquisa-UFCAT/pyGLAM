```python
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements_env.txt
poetry publish --build
git add . && git commit -m "fix: update requirements" && git tag "0.1.1" && git push && git push origin "0.1.1"
poetry version patch (vai de 0.1.1 para 0.1.2)
poetry version minor (vai de 0.1.1 para 0.2.0)
poetry version major (vai de 0.1.1 para 1.0.0)
./release.sh "fix: update requirements"
python release.py "atualiza dependencias do scipy e pandas"
```
