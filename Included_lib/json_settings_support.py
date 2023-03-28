import json


class JsonSettingSupport:
    def __init__(self):
        pass

    def create_file(self, _path_file, _template):
        pass

    @staticmethod
    def save_file(_path_file, _data_to_save):
        """
        Metoda zapisu danych do pliku typu json.

        :param _path_file: Lokalizacja pliku na dysku.
        :type _path_file: str
        :param _data_to_save: Dane do zapisu.
        :type _data_to_save: dict
        :return Status zapisanego pliku:
        :rtype: bool
        """

        try:
            with open(_path_file, 'w') as f:
                json.dump(_data_to_save, f)
            return True
        except PermissionError as error:
            print("Zla lokalizacja pliku '{}' ".format(_path_file))
            print(str(error))
            return False

    @staticmethod
    def load_file(_path_set_file):
        """
        Metoda odczytania danych z pliku typu json.

        :param _path_set_file: Lokalizacja pliku na dysku.
        :type _path_set_file: str
        :return Zawartość odczytanego pliku:
        :rtype: dict
        """
        try:
            f = open(_path_set_file)
            _data = json.load(f)
            f.close()
            return _data
        except FileNotFoundError as file_error:
            print("Plik {} nie istnieje".format(_path_set_file))
            print(str(file_error))
            return None
        except json.decoder.JSONDecodeError as file_error:
            print("Zły format pliku lub nie mozna odczytać danych")
            print(str(file_error))
            return None

    def load_parameters(self):
        pass

    def save_parameters(self):
        pass

    def change_parameters(self):
        pass

    def delete_file(self):
        pass

    def clear_file(self):
        pass


def main():
    # print(type(JsonSetting.load_file("settings/global.json")))
    set_json = JsonSettingSupport.load_file("settings/global.json")
    print(set_json)
    # JsonSetting.save_file("settings/", set_json)


if __name__ == '__main__':
    main()
