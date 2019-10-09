# coding: utf-8
import sys
import json
import logging
import tornado
import tornado.web
import tornado.ioloop
from tornado.httpserver import HTTPServer

sys.path.append('/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/')

from sentence_fluency.predictor import predict


WEB_PORT = 1368


class MainHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def post(self, *args, **kwargs):
        try:
            request_json = json.loads(self.request.body)
            sentence = request_json.get('sentence', '')
            logging.info("sentence：{}".format(sentence))
            if sentence:
                result = predict(sentence)
                logging.info("result: {}".format(result))
                self.write({
                        'code': 200,
                        'is_fluency': result
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
    ("/sentence_fluency/", MainHandler)
])

if __name__ == '__main__':
    server = HTTPServer(application)
    server.bind(WEB_PORT)
    server.start(num_processes=1)  # redis_info.num_processes)
    print("server is running on {}".format(WEB_PORT))
    tornado.ioloop.IOLoop.current().start()



