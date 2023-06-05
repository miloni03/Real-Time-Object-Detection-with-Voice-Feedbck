from flask import Flask, render_template, request
import subprocess
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run_script", methods=["POST"])
def run_script():
    result = subprocess.run(['python', 'detector.py'], capture_output=True)
    if result.returncode == 0:
        return "Script executed successfully."
    else:
        return "An error occurred: " + result.stderr.decode()
if __name__ == "__main__":
    app.run()

