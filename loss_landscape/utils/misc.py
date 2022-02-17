from datetime import datetime

def now2str():
    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M%S")
    return now_str