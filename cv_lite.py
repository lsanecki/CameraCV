import socket


class CvCameraLite:
    """Klasa do obslugi zdalnej kamery wersja lite do wykożystania w testerach"""

    def __init__(self, hostname: str, port=5001):
        """Konstruktor klasy
                :param hostname: nazwa serwera lub adres ip serwera
                :type hostname: str
                :param port: numer portu
                :type port: int
        """

        self.port = port
        self.server_address = None
        self._prepare_address(hostname)

    def _prepare_address(self, address_ip_or_name_host: str):
        """pobranie adresu ip serwera
        :param address_ip_or_name_host: Adres ip serwera lub nazwa serwera
        :type address_ip_or_name_host: str
        """

        if address_ip_or_name_host.split('.')[0].isdigit():  # sprawdz czy jest to adres ip
            address_ip_or_name_host = socket.gethostbyname(address_ip_or_name_host)
            self.server_address = address_ip_or_name_host
        else:
            address_ip_or_name_host = socket.gethostbyname(address_ip_or_name_host + '.local')
            self.server_address = address_ip_or_name_host

    def send_message_to_device(self, msg_to_send: str):
        """
        Wysłanie wiadomości do urządzenie poprzez protokul TCP/IP
        :param msg_to_send: Treść widomości do wyslania
        :type msg_to_send: str
        :return: Odebrana wiadomość od urządzenia/serwera
        :rtype: bytearray
        """

        client_socket = self._send_data_to_device(msg_to_send)
        recv_data = client_socket.recv(4096)
        client_socket.close()
        print('Received', recv_data)
        return recv_data

    def _send_data_to_device(self, data):
        """

        :param data:
        :return:
        """
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = None
        while result is None:  # reconnects if connection refused
            try:
                client_socket.connect((self.server_address, self.port))  # a tuple
                client_socket.sendall(data.encode('utf-8'))
            except OSError as e:
                print(e)
            else:
                break
        return client_socket


def main():
    barcode_reader = CvCameraLite("cvcamera", 5001)
    msg = "kod_data_matrix#300#500#550#750"
    recv_data = barcode_reader.send_message_to_device(msg)
    print(recv_data)


if __name__ == '__main__':
    main()
