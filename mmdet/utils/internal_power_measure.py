# Functions for measuring power consumption with internal power meters

import threading
import time
import json
import statistics
from struct import pack


class PowerMeasureInternal:
    def __init__(self):
        self.thread = threading.Thread(target=self._run_measure)
        self.finished = False
        self.mutex = threading.Lock()

        self.result = None

    def _run_measure(self):
        last_time = time.time()
        while True:
            with self.mutex:
                if self.finished:
                    break

            gpu = int(open('/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input').read())
            cpu = int(open('/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input').read())
            soc = int(open('/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power2_input').read())
            cv = int(open('/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power0_input').read())
            vddrq = int(open('/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power1_input').read())
            sys5v = int(open('/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power2_input').read())
            #print('GPU: ', gpu)
            #print('CPU: ', cpu)
            #print('SOC: ', soc)
            #print('CV: ', cv)
            #print('VDDRQ: ', vddrq)
            #print('SYS5V: ', sys5v)
            self.result.append(gpu + cpu + soc + cv + vddrq + sys5v)

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
    c = PowerMeasureInternal()
    c.start()
    time.sleep(10)
    print(c.stop())
