import socket
import sys
import json
import numpy as np
import struct 

port = 8220
address = ('', port)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(address)
server_socket.listen(5)

print "Listening for client . . ."
conn, address = server_socket.accept()
print "Connected to client at ", address

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame=json.loads(frame_data, fix_imports=True, encoding="bytes")
    finalFrame = np.asarray(frame["data"])
    np.save("otherCameraCenter",restored)
    conn.send("ack")
#pick a large output buffer size because i dont necessarily know how big the incoming packet is                                                    
'''i=0;
    output = conn.recv(2048);
    if output.strip() == "disconnect":
        conn.close()
        sys.exit("Received disconnect message.  Shutting down.")
        conn.send("dack")
	break
    elif output:
	print "Message received from client:"
        #print json.loads(output)
	json_load = json.loads(output)
	restored = np.asarray(json_load["data"])
	#print(restored)
	np.save("otherCameraCenter",restored)
        conn.send("ack")
        i=i+1'''                     
