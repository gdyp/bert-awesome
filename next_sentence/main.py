# coding: utf-8
import json
import logging
import tornado
import tornado.web
import tornado.ioloop
from tornado.httpserver import HTTPServer

from next_sentence.predict import predict

WEB_PORT = 1234
CHECKPOINT_DIR = '/data/gump/experiment/20190226/2019_02_21_21_06_46/'


class MainHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def post(self, *args, **kwargs):
        try:
            request_json = json.loads(self.request.body)
            text_a = request_json.get('text_a')
            text_b = request_json.get('text_b')
            logging.info("text_a：{}, text_b".format(text_a, text_b))
            if text_a and text_b:
                result = predict(text_a, text_b)

                self.write({
                        'code': 200,
                        'reasonable': result
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
    ("/answer_reasonable/", MainHandler)
])

if __name__ == '__main__':
    server = HTTPServer(application)
    server.bind(WEB_PORT)
    server.start(num_processes=1)  # redis_info.num_processes)
    print("server is running on {}".format(WEB_PORT))
    tornado.ioloop.IOLoop.current().start()



