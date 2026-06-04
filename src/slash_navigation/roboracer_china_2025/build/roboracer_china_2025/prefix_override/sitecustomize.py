import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jetson/slash_ws/src/slash_navigation/roboracer_china_2025/install/roboracer_china_2025'
