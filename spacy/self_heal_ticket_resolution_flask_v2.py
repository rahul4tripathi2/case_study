import flask
from flask import Flask, request

from flask import jsonify
from self_heal_ticket_resolution_v2 import similar_resolution


app = flask.Flask(__name__)


@app.route("/ticket_resolution", methods=['POST'])
def get_similar_ticket():
    print('suggesting different resolution for same problem')
    input_query = request.get_json()
    problem = input_query.get("problem")
    result = similar_resolution(problem)
    response = jsonify(result)

    return response


if __name__ == "__main__":
    app.run()

