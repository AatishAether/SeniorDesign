import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 55555))
server.listen()

client_socket, client_address = server.accept()

file = open("C:\\Users\\david\\Desktop\\Test\\newImage.png", "wb")
image_chunk = client_socket.recv(4096)

while image_chunk:
    file.write(image_chunk)
    image_chunk = client_socket.recv(4096)
file.close()
