import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 55555))

file = open("C:\\Users\\david\\Desktop\\Test\\img.png", 'rb')
image_data = file.read(4096)

while image_data:
    client.send(image_data)
    image_data = file.read(4096)

file.close()
client.close()
