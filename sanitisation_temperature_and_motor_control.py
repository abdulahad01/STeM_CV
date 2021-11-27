from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
from time import sleep

temp=0
temp_ary=[]
sani_valv=25
motor_r_o=27
motor_r_c=24
kill_start=23
kill_stop=22
ir_sen_state=4

GPIO.setmode(GPIO.BCM) # Broadcom pin-numbering scheme
GPIO.setwarnings(False)
GPIO.setup(ir_sen_state, GPIO.IN) # ir pin set as input w/ pull-up
GPIO.setup(kill_stop, GPIO.IN)#kill_switch for stoping the opening flap
GPIO.setup(kill_start, GPIO.IN)#kill_switch for stoping the closing flap
GPIO.setup(sani_valv, GPIO.OUT)#sanitizer valve activation (signal to MOSFET)
GPIO.setup(motor_r_o, GPIO.OUT)#flap opening command intiating pin
GPIO.setup(motor_r_c, GPIO.OUT)#flap closing command intiating pin

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
flag =0
# for testing purpose
#####################################
count=1
GPIO.output(motor_r_o,0)
GPIO.output(motor_r_c,0)
#####################################

while(1):
    if(GPIO.input(ir_sen_state)==0):
        #print(sensor.get_ambient())
        for i in range(100):
            temp= float(sensor.get_object_1())
            temp_ary.append(temp)
            print(max(temp_ary))
        GPIO.output(sani_valv,1)
        sleep(0.2)#delay in seconds(sanitiser dispensing)
        GPIO.output(sani_valv,0)
        sleep(0.5)
        if temp<=36.5:
            flag=1
            motor_r_state=1#if motor_r_state==1 flap in motion otherwise not in motion
        else:
            flag=0
            motor_r_state=0
        
        #mask detection to be initiated here
        
        #opening the flap
        while(GPIO.input(kill_stop)==1 and count !=100):
            GPIO.output(motor_r_o,motor_r_state)#initiated command to rod (motor) to open the way
            sleep(0.1)
            count=count+1# for testing
            
        GPIO.output(motor_r_o,0)
        
        #closing the flap
        count=0# for testing
        while(GPIO.input(kill_start)==1 and count !=10):
            GPIO.output(motor_r_c,motor_r_state)#initiated command to rod (motor) to open the way
            sleep(0.1)
            count=count+1# for testing
            
        GPIO.output(motor_r_c,0)
        print("Individual has been satisfied with all the parametrs")#simply for rasam :) heheheheee