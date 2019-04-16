# -*- coding: UTF-8 -*-
# filename: test date: 2019/1/20 15:45  
# author: FD 
import socket
sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect(('172.17.2.146',3000))
sock.send(b'open')
sock.close()