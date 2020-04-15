import socket
from concurrent.futures import ProcessPoolExecutor


def run_client(i):
    sock = socket.socket()
    sock.connect(('', 12345))

    with sock:
        print(f"client{i}: connect")
        sock.send("Hello world!".encode())
        print(f"client{i}: send")
        answer = sock.recv(1024)
        print(f"client{i}: answer = {answer.decode()}")


with ProcessPoolExecutor() as executor:
    executor.map(run_client, list(range(6)))
