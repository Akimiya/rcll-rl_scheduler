#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import socket
import time
import struct
import traceback
import threading
import datetime as dt
import numpy as np
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
#            "LogMessage",
            "BeaconSignal",
            "MachineReportInfo", # Not supported
#            "GameState" # we are RUNNING and in PRODUCTION anyway
            ]

def create_order(fill=False, amount=False, compet=False, window=False):
    # enum bases red = 1, black, silver
    base = int(np.random.uniform(1, 4))
    
    # number of rings (no information on distribution)
    rnd = np.random.uniform()
    if rnd <= 0.5:
        num_rings = 1
    elif rnd <= 0.8:
        num_rings = 2
    elif rnd <= 1:
        num_rings = 3
    # enum rings blue = 1, green, oragne, yellow
    rings = [int(np.random.uniform(1, 5)) for _ in range(num_rings)] + [0] * (3 - num_rings)
    
    # enum caps black = 1, grey
    cap = int(np.random.uniform(1, 3))
    
    # number of requested products
    if amount == True:
        num_products = [int(np.random.uniform(1, 3))]
    else:
        num_products = [0] if fill else []
    
    # if order is competitive
    if compet == True:
        competitive = [int(np.random.uniform(0, 2))]
    else:
        competitive = [0] if fill else []
        
    # the delivery window
    minimum_window = 120
    if window == True:
        start = int(np.random.uniform(1, 1021 - minimum_window))
        end = int(np.random.uniform(start + minimum_window, 1021))
        delivery_window = [start, end]
    else:
        delivery_window = [0, 0] if fill else []
    
    return [base] + rings + [cap] + num_products + competitive + delivery_window

# specifies how to encode the specific field
def create_byte_form(field_size):
    global test
    test = 42
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

def connect_socket(s, exist=True):
    if exist:
        s.close()
    
    while True:
        try:
            # create socket and build connection
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP, TCP_PORT))
            print("NEW Connection started: \n{}\n\n".format(s))
            break
        except ConnectionRefusedError:
            print("Connection refused, trying again..")
            s.close()
            time.sleep(1)
    
    return s

def communicator(debug_output=False, log_=False):
    # work with global main variables
    global machines
    global orders
    global rings
    global s
    global running
    global pb_obj
    global game_time
    
    # create socket and build connection
    s = connect_socket(None, False)
    
    if log_:
        message_file = open("messages.log", "a")
    
    # reset to default values
    # TODO: has instructions field - do we need?
    machines = {"CS1": deepcopy(MACHINE),
                "CS2": deepcopy(MACHINE),
                "RS1": deepcopy(MACHINE),
                "RS2": deepcopy(MACHINE),
                "SS": deepcopy(MACHINE),
                "BS": deepcopy(MACHINE),
                "DS": deepcopy(MACHINE)}
    last_machines = {}
    last_orders = pb.OrderInfo_pb2.OrderInfo()
    
    rings = {1 : 0,
             2 : 0,
             3 : 0,
             4 : 0}
    last_rings = {}
    
    game_time = 0
    
    # manage unpacking messages
    data = b""
    while running:
        try:
            data += s.recv(BUFFER_SIZE)
            # while there is something processable inside the buffer
            have_data = True
        except OSError:
            print("Worker got OSError..")
            data = b""
            
        if not data and running:
            have_data = False
            print("Connection lost, trying to reconnect")
            s = connect_socket(s)
        
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
                    # flag to write message to log file
                    to_log = False
                    
                    # create the protobuf object of the appropriate type
                    pb_obj_constructor = getattr(getattr(pb, component + "_pb2"), component)
                    pb_obj = pb_obj_constructor() # a new object
                    read = pb_obj.ParseFromString(message["protobuf_msg"])
                    assert read == message["payload_size"] - 4
                
                    
                    # also filtering excessive messages, get 3 messages: 
                    # 1) without state and ringcolors, 2) without ringcolors, 3) complete??
                    if component == "MachineInfo":
                        for m in pb_obj.machines:
                            # filter just one activve team; here CYAN
                            if m.name[0] == "C" and m.state != "": 
                                mtype = m.name[2:] # string for type
#                                print("THE STATE:", m.state, type(m.state), m.state == "IDLE", m.state != "IDLE")
                                machines[mtype]["state"] = m.state
                                machines[mtype]["loaded"] = m.loaded_with
                                machines[mtype]["ring_colors"] = list(m.ring_colors)
                                # process the lights
#                                for x in m.lights:
#                                    # 0 is OFF, 1 is ON, 2 is BLINK => MachineDescripton.proto
#                                    machines[mtype]["lights"][x.color] = x.state
#                                if mtype == "RS1":
#                                    assert machines[mtype]["ring_colors"] != []
                            
                        outp = [[k, v] for k, v in machines.items() if k != "SS"] # just for debug
                        
                        if last_machines != machines:
                            to_log = True
                            last_machines = deepcopy(machines)
                            if debug_output:
                                print("We got a <{}> message:\n{}\n{}\n{}".format(component, outp[:2], outp[2:4], outp[4:]))
                    
                    elif component == "OrderInfo":
                        orders = pb_obj.orders # we can work with given struct here as need all param
                        
                        if last_orders != orders:
                            to_log = True
                            last_orders = deepcopy(orders)
                            if debug_output:
                                print("We got a <{}> message:\n{}".format(component, pb_obj))
                        
                    elif component == "GameState":
                        game_time = pb_obj.game_time.sec
#                        if debug_output:
#                            print("We got a <{}> message:\n{}\n{}".format(component, game_time, pb_obj))
                            
                    elif component == "RingInfo":
                        # TODO: technically only need to know once
                        for r in pb_obj.rings:
                            rings[r.ring_color] = r.raw_material
                        
                        if last_rings != rings:
                            to_log = True
                            last_rings = deepcopy(rings)
                            if debug_output:
                                print("We got a <{}> message:\n{}".format(component, rings))
                        
                    else:
                        print("We got a <{}> message:\n{}".format(component, pb_obj))
                        to_log = True
                    
                    if log_ and to_log:
                        message_file.write("----------------------------------------------------------\n{} - <{}> - {:03d}:\n{}".format(dt.datetime.now(), component, game_time, pb_obj))
                        message_file.flush()
        
                # remove proccessed message from the buffer
                data = data[pos:]
        except Exception as e:
            running = False
            print("HAD EXCEPTION:\n{}".format(e))
            print(traceback.format_exc())
            raise e

    if log_:
        message_file.close()
        
    print("Workerprocess closed")


def simulate_game():
    global workpieces
    global orders
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
#    sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    sock_tcp.connect((TCP_IP, TCP_PORT))
#    sock_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP don't need anymore
    
    send_pb_message(set_team_name0, s)
    send_pb_message(set_team_name1, s)
    send_pb_message(set_game_state, s)
    time.sleep(5) # TODO: make C++ load old setting or just ommit machine positions
    send_pb_message(set_game_phase, s)
    
    # GAME COMMUNICATION
        
#    add_base = pb.MachineCommands_pb2.MachineAddBase()
#    add_base.machine_name = "ASD"
    
#    prepare_RS = pb.MachineInstructions_pb2.PrepareInstructionRS()
#    prepare_RS.ring_color = 4
#    prepare_machine = pb.MachineInstructions_pb2.PrepareMachine()
#    prepare_machine.team_color = 1
#    prepare_machine.machine = "M-RS1"
#    prepare_machine.instruction_rs.ring_color = 3
    
    time.sleep(2)
    
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
    
    send_pb_message(prepare_machine, s)

    
    # messages supposedly sent by the machines
    workpiece_info = pb.WorkpieceInfo_pb2.Workpiece()
    wp_id = 0
    if need_base == pb.ProductColor_pb2.BASE_RED:
        wp_id = workpieces["id_red"]
        workpieces["id_red"] += 1
    elif need_base == pb.ProductColor_pb2.BASE_BLACK:
        wp_id = workpieces["id_black"]
        workpieces["id_black"] += 1
    elif need_base == pb.ProductColor_pb2.BASE_SILVER:
        wp_id = workpieces["id_silver"]
        workpieces["id_silver"] += 1
    
    workpiece_info.id = wp_id
    workpiece_info.at_machine = "C-BS"
    workpiece_info.base_color = need_base
    # optional
    workpiece_info.team_color = 0
    
    send_pb_message(workpiece_info, s)


# TODO: The RefBox may or may not support intermediate points
#       IT MAY DEPEND ON (workpiece-tracking (enabled ?tracking-enabled)) IS ENABLED

# TODO: In real robocup have fall-back strategy if RL bugs out (e.g. machine broken needed)

# TODO: seems like the RefBox has a MOCKUP mode but it is not properly working and the machines do get timeouts

if __name__ == "__main__":
    global machines
    global orders
    global rings
    global s
    global running
    global workpieces
    
    machines = {}
    orders = {}
    rings = {}
    s = None
    running = True
    workpieces = {"id_red" : 1001,
                 "id_black" : 2001,
                 "id_silver" : 3001,
                 "id_clear" : 4001}
    
    # start the thread handling current data
    thr = threading.Thread(target=communicator, args=(True, True,))
    thr.start()
    
    try:
        while True:
            time.sleep(5)
#            print("MT orders:", orders)
    except KeyboardInterrupt:
        running = False
        s.close()
        print("Main got KeyboardInterrupt, exiting..")

    # thread should be closing due to socket closed
    thr.join()
    print("Main closed")