from flask import Flask, url_for, request, render_template

app = Flask(__name__)

from markupsafe import escape

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template("hello.html", name=name)

with app.test_request_context('/hello', method='POST'):
    assert request.path == "/hello"
    assert request.method == "POST"