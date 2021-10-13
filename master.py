from smbus import SMBus
from prototypeDetector import *

#bus address
addr = 0x8

#indicates /dev/ic2-1
bus = SMBus(1)


numb =1

print('Enter 1 for ON or 0 for OFF')

while numb ==1:

    isMaskSet = main()

    ledstate = isMaskSet
    
    if ledstate == '1':
        bus.write_byte(addr, 0x1)
    elif ledstate == '0':
        bus.write_byte(addr, 0x0)
    else:
        numb = 0
        
