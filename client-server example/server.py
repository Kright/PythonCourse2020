import socket
import time

from concurrent.futures import ThreadPoolExecutor

server_post = 12345

socket = socket.socket()
socket.bind(('', server_post))

print(f"server: run on port {server_post}")

socket.listen()


def process_connection(conn, addr):
    print(f"server: connection from {addr}")

    with conn:
        data = conn.recv(1024)
        time.sleep(3)
        conn.send(data)
        print("server: connection finished")


with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        conn, addr = socket.accept()
        executor.submit(process_connection, conn, addr)
