from datetime import timedelta
import time
def get_start_time():
    start_time = time.time()
    return start_time

def print_time_usage(start_time):
    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
