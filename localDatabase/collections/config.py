
# USERNAME = "jesus"
# PASSWORD = "Sm7p2rM0xc5ZRNbl"
DATABASE = "OptimizationAutonomousVehicles"
# IP = "172.24.192.1" # echo $(ip route list default | awk '{print $3}')
IP = "127.0.0.1" # echo $(ip route list default | awk '{print $3}')
PORT = "27017"
SERVER_URL = f"mongodb://" + IP + ":" + PORT