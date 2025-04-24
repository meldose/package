from neurapy.robot import Robot
import sys

r = Robot()
r.set_mode("Automatic")


io_get = r.io("get", io_name = "TDI_0")
# print(io_get)

io_set = r.io("set",io_name="DO_Array",target_value=[1,1,1,0,0,0,0,0]) # setting the tool digital outputs

# io_set_= r.io("set", io_name = "DO_1", target_value = True)


def set_signal_light(red, green, blue):
    if (red == 1):
        io_set_= r.io("set", io_name = "DO_0", target_value = True)
    else:
        io_set_= r.io("set", io_name = "DO_0", target_value = False)

    if (green == 1):
        io_set_= r.io("set", io_name = "DO_2", target_value = True)
    else:
        io_set_= r.io("set", io_name = "DO_2", target_value = False)

    if (blue == 1):
        io_set_= r.io("set", io_name = "DO_1", target_value = True)
    else:
        io_set_= r.io("set", io_name = "DO_1", target_value = False)

#sys.exit("no trafic light")
#set_signal_light(red="DO_0",green="DO_2",blue="DO_1")
