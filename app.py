import os
from dotenv import load_dotenv

# from langchain.agents import create_csv_agent
# from langchain_experimental.agents import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent


# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

from flask import Flask, jsonify, request
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Read OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY must be set in .env file")

@app.route('/predict', methods=['POST'])
def predict():
    
    file = request.files['file']

    file.save(file.filename)

    agent = create_csv_agent(OpenAI(api_key=openai_api_key, temperature=0), file.filename, verbose=True)

    print(agent.agent.llm_chain.prompt.template)

    question = request.form['question']

    result = agent.run(question)

    os.remove(file.filename)

    return jsonify({'result': result})

if __name__ == '__main__':
  app.run(debug=True)

"""
import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})  # Allow requests from localhost:3000

# Read OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY must be set in .env file")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file.save(file.filename)

    agent = create_csv_agent(OpenAI(api_key=openai_api_key, temperature=0), file.filename, verbose=True)

    print(agent.agent.llm_chain.prompt.template)

    question = request.form['question']

    result = agent.run(question)

    os.remove(file.filename)

    return jsonify({'result': result}), 200, {'Access-Control-Allow-Origin': 'http://localhost:3000'}

if __name__ == '__main__':
    app.run(debug=True)

"""

# import os
# from dotenv import load_dotenv
# # from langchain_experimental.agents import create_csv_agent
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
# from langchain_openai import OpenAI
# from flask import Flask, jsonify, request
# from flask_cors import CORS

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)
# CORS(app, resources={r"/predict": {"origins": "*"}})  # Allowing all origins for testing purposes

# # Read OpenAI API key from environment variable
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if openai_api_key is None:
#     raise ValueError("OPENAI_API_KEY must be set in .env file")

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['file']
#     file.save(file.filename)

#     agent = create_csv_agent(OpenAI(api_key=openai_api_key, temperature=0), file.filename, verbose=True)

#     print(agent.agent.llm_chain.prompt.template)

#     question = request.form['question']

#     result = agent.run(question)

#     os.remove(file.filename)

#     response = jsonify({'result': result})
#     response.headers.add('Access-Control-Allow-Origin', '*')  # Allowing all origins for testing purposes

#     return response

# if __name__ == '__main__':
#     app.run(debug=True)
