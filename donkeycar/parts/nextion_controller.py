"""
Nextion Display Controller
"""
import time
import nextion_lib as nxlib

class NextionController(object):
    '''
    A part to communicate with a Nextion Display over a uart. The 
    display is being used as a vehicle console
    '''
    def __init__(self):
        print("Starting Nextion Controller")
        ######### make connection to serial UART to read/write NEXTION
        self.ser = nxlib.ser
        self.recording = False
        # Ensure that the display is on page 2 
        nxlib.nx_setcmd_1par(self.ser, 'page', 2)
        nxlib.nx_setText(self.ser, 2,4,'Ok')
        self.running = True

    def poll(self):
            EndCom = "\xff\xff\xff"
            look_touch = 0.5  # in seconds
            print("detecting serial every {} second(s) ...".format(look_touch))
            while True:
                try:
                    touch=self.ser.read_until(EndCom)
                    if  hex(touch[0]) == '0x65':  #  touch event. If it's empty, do nothing
                        pageID_touch = touch[1]
                        compID_touch = touch[2]
                        event_touch = touch[3]
                        print("page= {}, component= {}, event= {}".format(pageID_touch,compID_touch,event_touch))
                        if (pageID_touch, compID_touch) == (2, 3):  # Record toggle button pressed
                            state = nxlib.nx_getValue(self.ser, 2, 3)
                            if (state == 0):
                                self.recording = False
                                nxlib.nx_setText(self.ser, 2,4,'Off')
                            else:
                                self.recording = True
                                nxlib.nx_setText(self.ser, 2,4,'On')
                    sleep(look_touch)  ### timeout the bigger the larger the chance of missing a push
                except:
                    pass
    
    def update(self):
        while self.running:
            self.poll()
    
    def run(self):
        self.poll()
        return self.run_threaded()

    def run_threaded(self):
        return self.recording
