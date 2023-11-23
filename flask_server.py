# from flask import Flask

# app = Flask(__name__)

# @app.route('/flask', methods=['GET'])
# def index():
#     return "Flask server"

# if __name__ == "__main__":
#     app.run(port=5000, debug=True)


# from viaduc import Viaduc

# app = Flask(__name__)

# @app.route('/get_gif')
# def get_gif():
#     # Load .npy file
#     # get from local storage
#     data = np.load('path/to/your/file.npy')

#     # turn into base64

#     im = []
#     image_array = np.load('data/0010.npy')
#     for i in range(len(image_array)):
#         im.append(Image.fromarray(image_array[i].astype('uint8')))



#     buffer = io.BytesIO()
#     im[0].save(buffer, format='GIF', save_all=True, append_images=im[1:], optimize=False, duration=200, loop=0)
#     buffer.seek(0)
#     data_uri = base64.b64encode(buffer.read()).decode('ascii')



#     # Convert to a format that can be easily transmitted
#     data_json = data.tolist()

#     return jsonify(data=data_json)

# if __name__ == '__main__':
#     app.run()








from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from flask_restful import Resource
import numpy as np
import base64
import io
from PIL import Image



# Import your python module containing the script
import public.createGif as scripts

app = Flask(__name__)
api = Api(app)
CORS(app, origins=['https://cmpt340-project-758b976dd842.herokuapp.com/'])
parser = reqparse.RequestParser()
parser.add_argument('text')

@app.route('/createGIF')
def createGIF():
    # Load .npy file
    # get from local storage
    data = np.load('path/to/your/file.npy')

    # turn into base64

    im = []
    image_array = np.load('data/0010.npy')
    for i in range(len(image_array)):
        im.append(Image.fromarray(image_array[i].astype('uint8')))



    buffer = io.BytesIO()
    im[0].save(buffer, format='GIF', save_all=True, append_images=im[1:], optimize=False, duration=200, loop=0)
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode('ascii')



    # Convert to a format that can be easily transmitted
    data_json = data.tolist()

    return "hi"
    return jsonify(data=data_json)
    # return jsonify({'prediction': prediction_result})

class YourClass(Resource):
    def post(self):
        args = parser.parse_args()
        # Invoke your text processing script here
        processed_text = scripts.text_processor(args['text'])
        response = {'data': processed_text}
        return response, 200

# This is where the routing is specified
api.add_resource(YourClass, '/your_api_endpoint')

if "__name__" == "__main__":
    app.run(host='https://cmpt340-project-758b976dd842.herokuapp.com/')



