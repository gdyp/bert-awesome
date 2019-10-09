# coding: utf-8
import os
import json
import logging
import tornado
import tornado.web
import tornado.ioloop
from tornado.httpserver import HTTPServer

from extract_features.sentnce_embedding import SentenceEmbedding

WEB_PORT = 1234

logging.basicConfig(level="INFO")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class MainHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def post(self, *args, **kwargs):
        try:
            request_json = json.loads(self.request.body)
            sentence = request_json.get('sentence', '')
            logging.info("sentence：{}".format(sentence))
            if sentence:
                result = sentence_embedding.sentence_embedding(sentence)
                # logging.info("result: {}".format(result))
                self.write({
                        'code': 200,
                        'sentence_embedding': result
                    })
            else:
                self.write({
                    'code': 300,
                    'message': 'can not get sentence'
                })
        except Exception as e:
            self.write({
                'code': 500,
                'message': e
            })
            logging.info(e)


application = tornado.web.Application([
    ("/sentence_embedding/", MainHandler)
])

if __name__ == '__main__':
    sentence_embedding = SentenceEmbedding()
    server = HTTPServer(application)
    server.bind(WEB_PORT)
    server.start(num_processes=1)  # redis_info.num_processes)
    print("server is running on {}".format(WEB_PORT))
    tornado.ioloop.IOLoop.current().start()



