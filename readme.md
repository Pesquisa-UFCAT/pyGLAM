python -m venv myenv
myenv\Scripts\activate
pip install -r requirements_env.txt
poetry publish --build
git add . && git commit -m "fix: update requirements" && git tag "0.1.1" && git push && git push origin "0.1.1"