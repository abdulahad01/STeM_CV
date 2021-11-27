from smbus2 import SMBus
from mlx90614 import MLX90614

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
temp_ary=[]

while(True):
    temp=[]
    for i in range(100):
        temp= float(sensor.get_object_1())
        temp_ary.append(temp)
        print(max(temp_ary))
