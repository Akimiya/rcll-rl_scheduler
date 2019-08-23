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
#    [key for key, value in COMPONENTS.items() if value == 'LogMessage'][0]
    
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
            

    if s.type == socket.SOCK_STREAM:
        ret = s.send(msg_bytes)
    else:
        ret = s.sendto(msg_bytes, (TCP_IP, 4446))
    
    return ret


def simulate_game():
    # TODO: currently we just assume that message arrived - check that?
    
    # start setting teams
    set_team_name0 = pb.GameInfo_pb2.SetTeamName()
    set_team_name0.team_name = "Carologistics"
    set_team_name0.team_color = 0
    set_team_name1 = pb.GameInfo_pb2.SetTeamName()
    set_team_name1.team_name = "GRIPS"
    set_team_name1.team_color = 1
    
    # set gamestate to needed phase & state
    set_game_phase = pb.GameState_pb2.SetGamePhase()
    set_game_phase.phase = pb.GameState_pb2.GameState.PRODUCTION
    
    set_game_state = pb.GameState_pb2.SetGameState()
    set_game_state.state = pb.GameState_pb2.GameState.RUNNING
    
        
    # send the above
    sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_tcp.connect((TCP_IP, TCP_PORT))
#    sock_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP don't need anymore
    
    send_pb_message(set_team_name0, sock_tcp)
    send_pb_message(set_team_name1, sock_tcp)
    send_pb_message(set_game_state, sock_tcp)
    time.sleep(5) # TODO: make C++ load old setting or just ommit machine positions
    send_pb_message(set_game_phase, sock_tcp)
    
    # GAME COMMUNICATION
        
#    add_base = pb.MachineCommands_pb2.MachineAddBase()
#    add_base.machine_name = "ASD"
    
#    prepare_RS = pb.MachineInstructions_pb2.PrepareInstructionRS()
#    prepare_RS.ring_color = 4
#    prepare_machine = pb.MachineInstructions_pb2.PrepareMachine()
#    prepare_machine.team_color = 1
#    prepare_machine.machine = "M-RS1"
#    prepare_machine.instruction_rs.ring_color = 3
    
    
    # want to extract prouct
    # take first for now...
    process_order = orders[0]
    # start with base
    need_base = process_order.base_color
    
    prepare_machine = pb.MachineInstructions_pb2.PrepareMachine()
    prepare_machine.team_color = 0
    prepare_machine.machine = "C-BS"
    
    # tell RS to output a ringcolor?
    prepare_machine.instruction_rs.ring_color = pb.ProductColor_pb2.RING_YELLOW
    # tell BS to offer a base at one of 2 gates
    prepare_machine.instruction_bs.side = pb.MachineInstructions_pb2.INPUT # (for now?) we dont care which side
    prepare_machine.instruction_bs.color = need_base
    
    send_pb_message(prepare_machine, sock_tcp)

    
    # messages supposedly sent by the machines
    workpiece_info = pb.WorkpieceInfo_pb2.Workpiece()
    workpiece_info.id = 1
    workpiece_info.at_machine = "C-BS"
    workpiece_info.base_color = need_base
    
    send_pb_message(workpiece_info, sock_tcp)


# TODO: The RefBox may or may not support intermediate points
#       IT MAY DEPEND ON (workpiece-tracking (enabled ?tracking-enabled)) IS ENABLED
    
# TODO: In real robocup have fall-back strategy if RL bugs out (e.g. machine broken needed)
    
    
# TODO: seems like the RefBox has a MOCKUP mode but it is not properly working and the machines do get timeouts

if __name__ == "__main__":
    # create socket and build connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    print("Connection started: \n{}\n\n".format(s))
    
    message_file = open("messages.log", "a")
    
    # work with global main variables (for now)
    global machines
    global orders
    global rings
    
    last_machines = {}
    
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
                        # reset to default values
                        # TODO: has instructions field - do we need?
                        machines = {"CS1": deepcopy(MACHINE),
                                    "CS2": deepcopy(MACHINE),
                                    "RS1": deepcopy(MACHINE),
                                    "RS2": deepcopy(MACHINE),
                                    "SS": deepcopy(MACHINE),
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
                        
                        outp = [[k, v] for k, v in machines.items() if k != "SS"] # just for debug
                        if last_machines == machines:
                            print("We got a <{}> message but nothing changed..")
                        else:
                            print("We got a <{}> message:\n{}\n{}\n{}".format(component, outp[:2], outp[2:4], outp[4:]))
                        last_machines = deepcopy(machines)
                    
                    elif component == "OrderInfo":
                        orders = pb_obj.orders # we can work with given struct here as need all param
                        print("We got a <{}> message:\n{}".format(component, pb_obj))
                    elif component == "GameState":
                        print("We got a <{}> message:\nphase={}".format(component, pb_obj.phase))
                    elif component == "RingInfo":
                        # TODO: technically only need to know once
                        rings = {1 : 0,
                                 2 : 0,
                                 3 : 0,
                                 4 : 0}
                        for r in pb_obj.rings:
                            rings[r.ring_color] = r.raw_material
                        print("We got a <{}> message:\n{}".format(component, rings))
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