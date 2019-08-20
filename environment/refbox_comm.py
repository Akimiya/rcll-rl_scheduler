#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import socket
import time
import struct
import traceback
import datetime as dt
from collections import OrderedDict
from copy import deepcopy # more performant then dict()

import protobuf.build as pb

# connection parameters
TCP_IP = "192.168.56.102"
TCP_PORT = 4444
BUFFER_SIZE = 1024

# layout of the packet, describing byte size; NOTE: Order matters!!
PACKET_LAYOUT = OrderedDict()
PACKET_LAYOUT["protocol_version"] = 1
PACKET_LAYOUT["cipher"] = 1
PACKET_LAYOUT["reserved1"] = 1
PACKET_LAYOUT["reserved2"] = 1
PACKET_LAYOUT["payload_size"] = 4
PACKET_LAYOUT["component_ID"] = 2
PACKET_LAYOUT["message_type"] = 2
PACKET_LAYOUT["protobuf_msg"] = None

# base layout of needed machine fields
MACHINE = {"state" : "",
           "loaded" : 0, # number of bases loaded into a CS
           "lights" : [0, 0, 0], # ordered list color-index: 0-> RED, 1-> YELLOW, 2-> GREEN
           "ring_colors" : [] # only set for RS
           }

# define message ID and type
COMPONENTS = {
        (2003, 1) : "LogMessage",
        (2000, 1) : "BeaconSignal",
        (2000, 2) : "AttentionMessage",
        (2000, 70) : "ExplorationSignal",
        (2000, 81) : "GameInfo",
        (2000, 20) : "GameState",
        (2000, 17) : "SetMachineState",
        (2000, 13) : "MachineInfo",
        (2000, 61) : "MachineReport",
        (2000, 41) : "OrderInfo",
        (2000, 110) : "RingInfo",
        (2000, 30) : "RobotInfo",
        (2000, 327) : "SimTimeSync",
        (2000, 3) : "VersionInfo",
        (2000, 56) : "WorkpieceInfo",
        (2000, 62) : "MachineReportInfo" # NOTE: as no file named this way message would not parse anyway!
        }

NOT_NEED = ["RobotInfo",
            "VersionInfo",
            "LogMessage",
            "BeaconSignal",
            "MachineReportInfo" # Not supported
            ]

# specifies how to encode the specific field
def create_byte_form(field_size):
    form = ">" # bytes in big-endian order
    if field_size == 1:
        form += 'B'
    elif field_size == 2:
        form += 'H'
    elif field_size == 4:
        form += 'I'
    return form

# pb_msg: message asa protobuf class
def send_pb_message(pb_msg, s):
#    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#    sock.sendto(msg_bytes, (TCP_IP, 4446))
    
#    [key for key, value in COMPONENTS.items() if value == 'LogMessage'][0]
    set_game_state = pb.GameState_pb2.SetGameState()
    set_game_state.state = pb.GameState_pb2.GameState.WAIT_START
    pb_msg = set_game_state
    
    
    # construct message
    message = dict(PACKET_LAYOUT)
    message["protocol_version"] = 2
    message["cipher"] = 0
    message["reserved1"] = 0
    message["reserved2"] = 0
    message["payload_size"] = pb_msg.ByteSize() + 4
    message["component_ID"] = pb_msg.COMP_ID
    message["message_type"] = pb_msg.MSG_TYPE
    message["protobuf_msg"] = pb_msg.SerializeToString()
    
    # encode long byte-string
    msg_bytes = b""
    for field_name, field_size in PACKET_LAYOUT.items():
        # function has the definition of the field sizes
        form = create_byte_form(field_size)
        if field_name == "protobuf_msg":
            msg_bytes += message[field_name]
        else:
            msg_bytes += struct.pack(form, message[field_name])
    
    return 0

if __name__ == "__main__":
    # create socket and build connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    print("Connection started: \n{}\n\n".format(s))
    
    message_file = open("messages.log", "a")
    
    # manage unpacking messages
    data = b""
    while True:
        data += s.recv(BUFFER_SIZE)
        
        # while there is something processable inside the buffer
        have_data = True
        try:
            # greater then 12 as we need at least 12 bytes for headers
            while have_data and len(data) >= 12:
                ##### extract one message from data
                message = dict(PACKET_LAYOUT)
                pos = 0 # position inside current data buffer
                for field_name, field_size in PACKET_LAYOUT.items():
                    
                    # as we are ordered the protobuf part is last
                    if field_name == "protobuf_msg":
                        #take the raw bytes
                        protobuf_size = message["payload_size"] - 4
                        # check if we have enough bytes for message
                        if protobuf_size + 12 > len(data):
                            have_data = False
                            break
                        message[field_name] = data[pos:pos + protobuf_size] # minus 4 for the header
                        pos += protobuf_size
                        continue
                    
                    # reconstructed byte part
                    recon = data[pos:pos + field_size]
                        
                    # distinguish format
                    form = create_byte_form(field_size)
                    
                    # convert to proper size
                    up = struct.unpack(form, recon)
                    assert len(up) == 1
                    message[field_name] = up[0]
        
                    pos += field_size # current position
                
                if have_data == False:
                    break
                
                ##### get the protobuf message object
                try:
                    component = COMPONENTS[(message["component_ID"], message["message_type"])]
                except KeyError as e:
                    print("We cound not find given component ID and message type combination!\n{}".format(message))
                    raise e
                
                ##### proccess incoming protobuff messages
                if component not in NOT_NEED:
                    # create the protobuf object of the appropriate type
                    pb_obj_constructor = getattr(getattr(pb, component + "_pb2"), component)
                    pb_obj = pb_obj_constructor() # a new object
                    read = pb_obj.ParseFromString(message["protobuf_msg"])
                    assert read == message["payload_size"] - 4
                
                    if component == "MachineInfo":
                        machines = {"SS": deepcopy(MACHINE),
                                    "CS1": deepcopy(MACHINE),
                                    "CS2": deepcopy(MACHINE),
                                    "RS1": deepcopy(MACHINE),
                                    "RS2": deepcopy(MACHINE),
                                    "BS": deepcopy(MACHINE),
                                    "DS": deepcopy(MACHINE)}

                        for m in pb_obj.machines:
                            # filter just one activve team; here CYAN
                            if m.name[0] == "C":
                                mtype = m.name[2:] # string for type
                                
                                machines[mtype]["state"] = m.state
                                machines[mtype]["loaded"] = m.loaded_with
                                machines[mtype]["ring_colors"] = list(m.ring_colors)
                                # process the lights
                                for x in m.lights:
                                    # 0 is OFF, 1 is ON, 2 is BLINK => MachineDescripton.proto
                                    machines[mtype]["lights"][x.color] = x.state
                        
#                            if m.name == "C-RS1":
                        print("We got a <{}> message:\n{}".format(component, machines))
                    
                    elif component == "OrderInfo":
                        orders = pb_obj.orders # we can work with givne struct here
                    elif component == "GameState":
                        print("We got a <{}> message:\nphase={}".format(component, pb_obj.phase))
                    elif component == "BeaconSignal":
                        assert False
                    else:
                        print("We got a <{}> message:\n{}".format(component, pb_obj))
#                time.sleep(1)
#                    message_file.write("----------------------------------------------------------\n{} - <{}>:\n{}".format(dt.datetime.now(), component, pb_obj))
#                else:
#                    pb_obj_constructor = getattr(getattr(pb, component + "_pb2"), component)
#                    pb_obj = pb_obj_constructor() # a new object
#                    read = pb_obj.ParseFromString(message["protobuf_msg"])
#                    assert read == message["payload_size"] - 4
#                    message_file.write("----------------------------------------------------------\n{} - <{}>:\n{}".format(dt.datetime.now(), component, pb_obj))
                
        
                # remove proccessed message from the buffer
                data = data[pos:]
        except Exception as e:
            print("HAD EXCEPTION:\n{}".format(e))
            print(traceback.format_exc())
            raise e

    message_file.close()
    s.close()