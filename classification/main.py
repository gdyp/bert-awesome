# coding: utf-8
import json
import logging
import tornado
import tornado.web
import tornado.ioloop
from tornado.httpserver import HTTPServer

from predict import Predict

WEB_PORT = 1368
CHECKPOINT_DIR = '/data/gump/experiment/20190226/2019_02_21_21_06_46/'

predictor = Predict()


class MainHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def post(self, *args, **kwargs):
        try:
            request_json = json.loads(self.request.body)
            sentence = request_json.get('query', '')
            logging.info("query：{}".format(sentence))
            if sentence:
                result = predictor.predict(sentence)

                self.write({
                        'result': result
                    })
        except Exception as e:
            logging.info(e)


application = tornado.web.Application([
    ("/bert_classification/", MainHandler)
])

if __name__ == '__main__':
    server = HTTPServer(application)
    server.bind(WEB_PORT)
    server.start(num_processes=1)  # redis_info.num_processes)
    print("server is running on {}".format(WEB_PORT))
    tornado.ioloop.IOLoop.current().start()



