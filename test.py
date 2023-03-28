from Included_lib.barcode_reader import BarcodeReader


def main():
    barcode = BarcodeReader()
    print(barcode.camera.settings)


if __name__ == "__main__":
    main()
