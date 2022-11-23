from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

app = Flask(__name__)
api = Api(app)

class Api(Resource):
    def get(self):
        return { 'data': True }, 200
    def post(self):
        print(self)

class Teste(Resource):
    def post(self):
        # file = request.files['imagem'].filename
        # img = mpimg.imread(request.files['imagem'])
        # plt.imshow(img)
        # plt.show()
        img = Image.open(request.files['imagem'])
        print('before resize')
        print('width: ',img.width)
        print('height: ',img.height)

        print('after resize')
        img_resize = img.resize((28,28), Image.ANTIALIAS)
        print('width: ',img_resize.width)
        print('height: ',img_resize.height)
        # x = mpimg.imread(img_resize)
        plt.imshow(img_resize)
        plt.show()

        print(self)

api.add_resource(Api, '/is-alive')
api.add_resource(Teste, '/teste')

if __name__ == '__main__':
    app.run()  # run our Flask app