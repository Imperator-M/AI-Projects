# Ad-hoc garbage code for threads that can be killed
# From https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/

import threading
from threading import Thread, Event
import ctypes
import sys

class KillableThread(Thread):
    def __init__(self, sleep_interval=0, target=None, name=None, args=(), kwargs={}):
        super().__init__(None, target, name, args, kwargs)
        self._kill = Event()
        self._interval = sleep_interval

    def run(self):
        while True:
            # Call custom function with arguments
            self._target(*self._args)

            # If no kill signal is set, sleep for the interval,
            # If kill signal comes in while sleeping, immediately
            #  wake up and handle
            is_killed = self._kill.wait(self._interval)
            if is_killed:
                break

        print("Killing Thread")

    def kill(self):
        self._kill.set()

class thread_with_exception(threading.Thread):
    def __init__(self, target, args):
        threading.Thread.__init__(self)
        self.func = target
        self.args = args
             
    def run(self):
 
        # target function of the thread class
        try:
            while True:
                self.func(*self.args)
        finally:
            pass
          
    def get_id(self):
 
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
  
    def raise_exception(self):
        raise Exception

import sys
import trace
import threading
import time
class thread_with_trace(threading.Thread):
  def __init__(self, *args, **keywords):
    threading.Thread.__init__(self, *args, **keywords)
    self.killed = False
 
  def start(self):
    self.__run_backup = self.run
    self.run = self.__run     
    threading.Thread.start(self)
 
  def __run(self):
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup
 
  def globaltrace(self, frame, event, arg):
    if event == 'call':
      return self.localtrace
    else:
      return None
 
  def localtrace(self, frame, event, arg):
    if self.killed:
      if event == 'line':
        raise SystemExit()
    return self.localtrace
 
  def kill(self):
    self.killed = True