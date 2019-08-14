#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import socket
import time
import struct
import traceback
import datetime as dt
from collections import OrderedDict

import protobuf.build as pb

# connection parameters
TCP_IP = "192.168.56.102"
TCP_PORT = 4444
BUFFER_SIZE = 1024

# layout of the packet; NOTE: Order matters!!
PACKET_LAYOUT = OrderedDict()
PACKET_LAYOUT["protocol_version"] = 1
PACKET_LAYOUT["cipher"] = 1
PACKET_LAYOUT["reserved1"] = 1
PACKET_LAYOUT["reserved2"] = 1
PACKET_LAYOUT["payload_size"] = 4
PACKET_LAYOUT["component_ID"] = 2
PACKET_LAYOUT["message_type"] = 2
PACKET_LAYOUT["protobuf_msg"] = None

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
        (2000, 56) : "WorkpieceInfo"
        }
    

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
                    form = ">" # bytes in big-endian order
                    if field_size == 1:
                        form += 'B'
                    elif field_size == 2:
                        form += 'H'
                    elif field_size == 4:
                        form += 'I'
                    
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
                
                # create the protobuf object of the appropriate type
                pb_obj_constructor = getattr(getattr(pb, component + "_pb2"), component)
                pb_obj = pb_obj_constructor() # a new object
                read = pb_obj.ParseFromString(message["protobuf_msg"])
                assert read == message["payload_size"] - 4
                
                ##### proccess protobuff message
                # TODO: process
                print("We got a <{}> message".format(component, pb_obj))
#                time.sleep(1)
                message_file.write("----------------------------------------------------------\n{} - <{}>:\n{}".format(dt.datetime.now(), component, pb_obj))
                
        
                # remove proccessed message from the buffer
                data = data[pos:]
        except Exception as e:
            print("HAD EXCEPTION:\n{}".format(e))
            print(traceback.format_exc())

    message_file.close()
    s.close()