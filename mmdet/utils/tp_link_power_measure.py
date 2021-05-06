# Functions for measuring power consumption with TP-Link smart power plugs (HS-110)

import threading
import time
import socket
import json
import statistics
from struct import pack


# cryptographic functions src: https://raw.githubusercontent.com/softScheck/tplink-smartplug/master/tplink_smartplug.py


def _encrypt(string):
    key = 171
    result = pack('>I', len(string))
    for i in string:
        a = key ^ ord(i)
        key = a
        result += bytes([a])
    return result


def _decrypt(string):
    key = 171
    result = ""
    for i in string:
        a = key ^ i
        key = i
        result += chr(a)
    return result


class PowerMeasure:
    def __init__(self, ip):
        self.thread = threading.Thread(target=self._run_measure)
        self.finished = False
        self.mutex = threading.Lock()

        self.ip = ip
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, 9999))

        self.result = None

    def _run_measure(self):
        last_time = time.time()
        while True:
            with self.mutex:
                if self.finished:
                    break

            self.socket.send(_encrypt('{"emeter":{"get_realtime":{}}}'))
            data = self.socket.recv(2048)

            data = json.loads(_decrypt(data[4:]))
            assert data['emeter']['get_realtime']['err_code'] == 0

            self.result.append(data['emeter']['get_realtime']['power_mw'])

            new_time = time.time()
            time.sleep(max(0.0, 0.5 - (new_time - last_time)))
            last_time = time.time()

    def start(self):
        self.result = []
        self.thread.start()

    def stop(self):
        with self.mutex:
            self.finished = True
        self.thread.join()
        mean_power = statistics.mean(self.result)
        return mean_power


if __name__ == '__main__':
    c = PowerMeasure('192.168.0.200')
    c.start()
    time.sleep(1000)
    print(c.stop())
