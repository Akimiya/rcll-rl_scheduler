#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import socket
import sys
import struct
from collections import OrderedDict

import protobuf.build as pb

# connection parameters
TCP_IP = "192.168.56.102"
TCP_PORT = 4444
BUFFER_SIZE = 1024 * 10

# layout of the packet; NOTE: Order matters!!
PACKET_LAYOUT = OrderedDict()
PACKET_LAYOUT["protocol_version"] = 1
PACKET_LAYOUT["cipher"] = 1
PACKET_LAYOUT["reserved1"] = 1
PACKET_LAYOUT["reserved2"] = 1
PACKET_LAYOUT["payload_size"] = 4
PACKET_LAYOUT["component_ID"] = 2
PACKET_LAYOUT["message_type"] = 2
PACKET_LAYOUT["protobuf"] = None



if __name__ == "__main__":
    # create socket and build connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    print("Connection started: \n{}\n\n".format(s))
    
    # manage unpacking messages
    while True:
        data = s.recv(BUFFER_SIZE)
        
        # extract one message from data
        message = dict(PACKET_LAYOUT)
        pos = 0 # position inside current data
        for field_name, field_size in PACKET_LAYOUT.items():
            
            # as we are ordered the protobuf part is last
            if field_name == "protobuf":
                #take the raw bytes
                protobuf_size = message["payload_size"] - 4
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

            pos += field_size

    

    team = pb_team.Team
    time = pb_time.Time()
    pose = pb_pose.Pose2D()
    beacon_signal = pb_beacon_signal.BeaconSignal()
    machine_info = pb_machine_info.MachineInfo()




try:
    while True:
        data = s.recv(BUFFER_SIZE)
        print("GOT: {}".format(data))
except KeyboardInterrupt:
    print("closing...")
s.close()

msgs = []
while True:
    msgs.append(s.recv(BUFFER_SIZE))

for m in msgs:
    print(m)
    time.sleep(1)

sock = s
sz = 0
while True:
    vbyte, = struct.unpack('b', sock.recv(1))
    sz = (vbyte << 7) + (vbyte & 0x7f)
    if not vbyte & 0x80:
        break
data = []
while sz:
    buf = sock.recv(sz)
    if not buf:
        raise ValueError("Buffer receive truncated")
    data.append(buf)
    sz -= len(buf)


machine_info.FromString(msgs[7])

