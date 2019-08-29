'''import socket, json, pickle

PORT = 12345

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("10.40.3.3", PORT))
#while True:

arr = ([1],[1])
data_string = pickle.dumps(arr)
s.send(data_string)
s.send(data_string)
s.send(data_string)
s.send(data_string)
s.close();'''
import socket,json
import time

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("10.40.3.3", 8220))
data =[1,1]
#for index in xrange(5):
while True:
    #data = "GET\nSONAR%d\n\n" % index
    print 'send to server: ',data
    client_socket.send(json.dumps(data))
    while client_socket.recv(2048) != "ack":
        print "waiting for ack"
    print "ack received!"

#send disconnect message                                                                                                                           
dmsg = "disconnect"
print "Disconnecting"
client_socket.send(dmsg)

client_socket.close()

