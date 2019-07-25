# -*- coding: UTF-8 -*-
# filename: dataserver date: 2018/11/21 15:23  
# author: FD 
import socketserver
import time
import threading


class FileServer(socketserver.BaseRequestHandler):
    def handle(self):
        filename = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        print('connected from zip file:', self.client_address)
        with self.request:
            with open(filename + ".zip", 'wb') as file:
                while True:
                    rdata = self.request.recv(1024)
                    if (len(rdata) == 0):
                        break
                    file.write(rdata)
        print('receive successfully!')


threading.Thread(
    target=lambda x: socketserver.ThreadingTCPServer(("0.0.0.0", 9999), FileServer).serve_forever(),args=(None,)).start()

commands = ['start', 'stop', 'next', 'send', 'remove', 'info']


class CommandServer(socketserver.BaseRequestHandler):
    def print_received(self):
        while (True):
            # try:
                print(self.request.recv(1).decode("utf-8"),end='')
            # except Exception:
            #     print()

    def handle(self):
        print('connected from:', self.client_address)
        threading.Thread(target=self.print_received).start()

        while (True):
            try:
                command = input("input command ( start 0, stop 1, next 2, send 3, remove 4, info 5): ")
                self.request.sendall((commands[int(command)] + "\n").encode())
            except Exception:
                print()


s2 = socketserver.ThreadingTCPServer(("0.0.0.0", 10000), CommandServer)
s2.serve_forever()
