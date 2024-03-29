#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:10:44 2017

@author: wroscoe

remotes.py

The client and web server needed to control a car remotely. 
"""


import os
import json
from json.decoder import JSONDecodeError
import time
import asyncio

import requests
from tornado.ioloop import IOLoop
from tornado.web import Application, RedirectHandler, StaticFileHandler, \
    RequestHandler

from tornado.httpserver import HTTPServer
import tornado.gen
import tornado.websocket
from socket import gethostname

from ... import utils


class RemoteWebServer():
    '''
    A controller that repeatedly polls a remote webserver and expects
    the response to be angle, throttle and drive mode. 
    '''
    
    def __init__(self, remote_url, connection_timeout=.25):

        self.control_url = remote_url
        self.time = 0.
        self.angle = 0.
        self.throttle = 0.
        self.mode = 'user'
        self.recording = False
        # use one session for all requests
        self.session = requests.Session()

    def update(self):
        '''
        Loop to run in separate thread the updates angle, throttle and 
        drive mode. 
        '''

        while True:
            # get latest value from server
            self.angle, self.throttle, self.mode, self.recording = self.run()

    def run_threaded(self):
        ''' 
        Return the last state given from the remote server.
        '''
        return self.angle, self.throttle, self.mode, self.recording

    def run(self):
        '''
        Posts current car sensor data to webserver and returns
        angle and throttle recommendations. 
        '''
        
        data = {}
        response = None
        while response is None:
            try:
                response = self.session.post(self.control_url,
                                             files={'json': json.dumps(data)},
                                             timeout=0.25)

            except requests.exceptions.ReadTimeout as err:
                print("\n Request took too long. Retrying")
                # Lower throttle to prevent runaways.
                return self.angle, self.throttle * .8, None

            except requests.ConnectionError as err:
                # try to reconnect every 3 seconds
                print("\n Vehicle could not connect to server. Make sure you've " +
                    "started your server and you're referencing the right port.")
                time.sleep(3)

        data = json.loads(response.text)
        angle = float(data['angle'])
        throttle = float(data['throttle'])
        drive_mode = str(data['drive_mode'])
        recording = bool(data['recording'])

        return angle, throttle, drive_mode, recording

    def shutdown(self):
        pass
    
    
class LocalWebController(tornado.web.Application):

    def __init__(self, port=8887, mode='user'):
        ''' 
        Create and publish variables needed on many of 
        the web handlers.
        '''

        print('Starting Donkey Server...', end='')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = mode
        self.recording = False
        self.port = port
        
        self.num_records = 0
        self.wsclients = []


        handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/wsDrive", WebSocketDriveAPI),
            (r"/video", VideoAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path}),
        ]

        settings = {'debug': True}
        super().__init__(handlers, **settings)
        print("... you can now go to {}.local:8887 to drive "
              "your car.".format(gethostname()))

    def update(self):
        ''' Start the tornado webserver. '''
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        IOLoop.instance().start()

    def run_threaded(self, img_arr=None, num_records=0):
        self.img_arr = img_arr
        self.num_records = num_records

        # Send record count to websocket clients
        if (self.num_records is not None and self.recording is True):
            if self.num_records % 10 == 0:
                for wsclient in self.wsclients:
                    wsclient.write_message(json.dumps({'num_records': self.num_records}))
        
        return self.angle, self.throttle, self.mode, self.recording
        
    def run(self, img_arr=None):
        self.img_arr = img_arr
        return self.angle, self.throttle, self.mode, self.recording

    def shutdown(self):
        pass


class DriveAPI(RequestHandler):

    def get(self):
        data = {}
        self.render("templates/vehicle.html", **data)

    def post(self):
        '''
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        '''
        data = tornado.escape.json_decode(self.request.body)
        self.application.angle = data['angle']
        self.application.throttle = data['throttle']
        self.application.mode = data['drive_mode']
        self.application.recording = data['recording']


class WebSocketDriveAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        # print("New client connected")
        self.application.wsclients.append(self)

    def on_message(self, message):
        data = json.loads(message)
        self.application.angle = data['angle']
        self.application.throttle = data['throttle']
        self.application.mode = data['drive_mode']
        self.application.recording = data['recording']

    def on_close(self):
        # print("Client disconnected")
        self.application.wsclients.remove(self)


class VideoAPI(RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle.
    '''

    async def get(self):

        self.set_header("Content-type",
                        "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:

            interval = .01
            if served_image_timestamp + interval < time.time() and \
                    hasattr(self.application, 'img_arr'):

                img = utils.arr_to_binary(self.application.img_arr)
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                served_image_timestamp = time.time()
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    pass
            else:
                await tornado.gen.sleep(interval)


class BaseHandler(RequestHandler):
    """ Serves the FPV web page"""
    async def get(self):
        data = {}
        await self.render("templates/base_fpv.html", **data)


class WebFpv(Application):
    """
    Class for running an FPV web server that only shows the camera in real-time.
    The web page contains the camera view and auto-adjusts to the web browser
    window size. Conjecture: this picture up-scaling is performed by the
    client OS using graphics acceleration. Hence a web browser on the PC is
    faster than a pure python application based on open cv or similar.
    """

    def __init__(self, port=8890):
        self.port = port
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')

        """Construct and serve the tornado application."""
        handlers = [
            (r"/", BaseHandler),
            (r"/video", VideoAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path})
        ]

        settings = {'debug': True}
        super().__init__(handlers, **settings)
        print("Started Web FPV server. You can now go to {}.local:{} to "
              "view the car camera".format(gethostname(), self.port))

    def update(self):
        """ Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        IOLoop.instance().start()

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr

    def run(self, img_arr=None):
        self.img_arr = img_arr

    def shutdown(self):
        pass


class WebConsole(tornado.web.Application):
    
    def __init__(self, cfg, port=8887):
        ''' 
        Provide for dynamic vehicle configuration setting. 
        '''

        print('Starting Console Server...', end='')

        self.port = port
        self.cfg = cfg
        self.AIPilot = cfg.AIPILOT
        self.RunState = 'Initializing'
        self.RunCmd = 'None'
        
        handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", ConsoleAPI),
        ]
        
        settings = {'debug': True}
        super().__init__(handlers, **settings)
        print("... you can now go to {}.local:8887 to control "
              "your car.".format(gethostname()))

    def update(self):
        ''' Start the tornado webserver. '''
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        IOLoop.instance().start()

    def run_threaded(self,RunState):
        if self.RunCmd == 'start':
            self.RunState = 'running'
        elif self.RunCmd == 'stop':
            self.RunState = 'ready'
        else:
            self.RunState = RunState
        return self.RunState
        
    def run(self,RunState):
        if self.RunCmd == 'start':
            self.RunState = 'running'
        elif self.RunCmd == 'stop':
            self.RunState = 'ready'
        else:
            self.RunState = RunState
        return self.RunState
    
    def shutdown(self):
        pass

class ConsoleAPI(RequestHandler):

    def get(self):
        # Set up response dictionary.
        self.response = dict()       
        self.response['AIPilot'] = self.application.AIPilot
        self.response['RunState'] = self.application.RunState
        output = json.dumps(self.response)
        self.write(output)

    def post(self):
        try:
            data = json.loads(self.request.body.decode('utf-8'))
            print('Got JSON data:', data)
            self.write({ 'got' : 'your data' })
            self.application.RunCmd = data['RunCmd']
            if self.application.RunCmd == 'start':
                self.application.RunState = 'running'
            elif self.application.RunCmd == 'stop':
                self.application.RunState = 'ready'
        except JSONDecodeError as e:
            print('Could not decode message',self.request.body)

