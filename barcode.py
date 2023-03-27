#!/usr/bin/env python3
import cv2
from pyzbar import pyzbar
import socket
from time import sleep, time
import socketserver
import numpy as np
from RPi import GPIO as GPIO
import os
import socket
from dmtx import *
from pylibdmtx.pylibdmtx import decode
import zxing
import pickle
import struct
########################################  V 2.0

#TODO: naprawic blad wyjatek wielkosci kodu zmienna org_list w pliku dmtx
#TODO: naprawić błąd nie kasuja sie dane po dekodowaniu kodu Datamatrix i cały czas odczytuje z tego samego zdjecia


class Barcode:
	def __init__(self) -> None:
		self.dmtx_lib = DMTX()
		self.width = 1280
		self.height = 1024

		self.HOST = ''  # Standard loopback interface address (localhost)
		self.PORT = 5001

		self.Selection = 7
		self.Enable1 = 11
		self.Enable2 = 12

		self.camera = cv2.VideoCapture(0)
		self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
		self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(self.Selection, GPIO.OUT,initial=GPIO.HIGH)
		GPIO.setup(self.Enable1, GPIO.OUT, initial=GPIO.HIGH)
		GPIO.setup(self.Enable2, GPIO.OUT, initial=GPIO.HIGH)

		self.zxing_reader = zxing.BarCodeReader()

	def zalacz_kamere(self, nr_cam):
			if nr_cam == 0:
					GPIO.output(self.Selection, GPIO.HIGH)
					GPIO.output(self.Enable1, GPIO.HIGH)
					GPIO.output(self.Enable2, GPIO.HIGH)
			elif nr_cam == 1:
					i2c = "i2cset -y 1 0x70 0x00 0x04"
					os.system(i2c)
					GPIO.output(self.Selection, GPIO.LOW)
					GPIO.output(self.Enable1, GPIO.LOW)
					GPIO.output(self.Enable2, GPIO.HIGH)
			elif nr_cam == 2:
					i2c = "i2cset -y 1 0x70 0x00 0x05"
					os.system(i2c)
					GPIO.output(self.Selection, GPIO.HIGH)
					GPIO.output(self.Enable1, GPIO.LOW)
					GPIO.output(self.Enable2, GPIO.HIGH)
			elif nr_cam == 3:
					i2c = "i2cset -y 1 0x70 0x00 0x06"
					os.system(i2c)
					GPIO.output(self.Selection, GPIO.LOW)
					GPIO.output(self.Enable1, GPIO.HIGH)
					GPIO.output(self.Enable2, GPIO.LOW)
			elif nr_cam == 4:
					i2c = "i2cset -y 1 0x70 0x00 0x07"
					os.system(i2c)
					GPIO.output(self.Selection, GPIO.HIGH)
					GPIO.output(self.Enable1, GPIO.HIGH)
					GPIO.output(self.Enable2, GPIO.LOW)

	def data_matrix(self, frame):
		img = self.dmtx_lib.grayscale_bin(frame.copy())
		if type(img) is np.ndarray:
			img, rectangle = self.dmtx_lib.remove_background(img)
			angle = rectangle[2] - 90
			if type(img) is np.ndarray:		
				rotated_img = self.dmtx_lib.rotate_image(img, angle)
				if type(rotated_img) is np.ndarray:	
					borderless_img = self.dmtx_lib.remove_white_border(rotated_img)
					if type(borderless_img) is np.ndarray:	
						dmtx_size_cadidates = self.dmtx_lib.detect_dmtx_size(borderless_img)
						if isinstance(dmtx_size_cadidates, list):
							if dmtx_size_cadidates:
								for pixel_size in dmtx_size_cadidates:
									self.dmtx_lib.dmtx_pixel_count = pixel_size
									regenerated_img = self.dmtx_lib.regenarate(borderless_img)
									if type(regenerated_img) is np.ndarray:
										rotated_regenerated_img = self.dmtx_lib.rotate_dmtx(regenerated_img)		
										if type(rotated_regenerated_img) is np.ndarray:
											swollen_dmtx = self.dmtx_lib.matrix_swell(rotated_regenerated_img, swell_factor=1)
											if type(swollen_dmtx) is np.ndarray:
												no_outline_img = self.dmtx_lib.remove_dmtx_outline(swollen_dmtx)
												if type(no_outline_img) is np.ndarray:			
													codewords_list = self.dmtx_lib.dmtx_to_codewords(no_outline_img)
													if isinstance(codewords_list, list):												
														decoded, errors = self.dmtx_lib.decode_RS(codewords_list)
														if errors == -1:
															message = 'error' + '#' + 'Za duzo bledow nie odczytam = '+ str(errors)	
															print(message)
															kod = bytes(message.encode('utf-8'))
														else:
															kod = self.dmtx_lib.convert_code(decoded)
															kod = bytes(kod.encode('utf-8'))
															print('Zdekodowano: ', kod)
													else:
														kod = codewords_list
												else:
													kod = no_outline_img
											else:
												kod = swollen_dmtx
										else:
											kod = rotated_regenerated_img														
									else:
										kod = regenerated_img																				
							else:
								message = 'error' + '#' + 'Lista rozmiarow pusta'
								print(message)
								kod = bytes(message.encode('utf-8'))
						else:
							kod = dmtx_size_cadidates							
					else:
						kod = borderless_img
				else:
					kod = rotated_img
			else:
				kod = img
		else:
			kod = img
		return kod

	def kod_kreskowy(self, frame):
		try:
				kod = pyzbar.decode(frame)
				kod = kod[0].data
				if kod == '':
					pass
		except:
				kod = bytes(1)
		print('Zdekodowano: ',kod)
		return kod

	def zapisz_zdjecie(self, frame, file_name):
		cv2.imwrite(str(file_name)+'.jpg',frame)

	def unique(self):
		now = time()
		now = str(int(now *1000))
		return now

	def zapisz_NAS(self, frame):
		file_name = self.unique()
		file_name = '/home/pi/NAS/'+str(socket.gethostname())+'_' + str(file_name)+'.webp'
		cv2.imwrite(file_name,frame, [cv2.IMWRITE_WEBP_QUALITY, 50])

	def zrob_zdjecie(self):
		ret, frame = self.camera.read()	  
		if ret:
			return frame
		else:
			message = 'error' + '#' + 'brak zdjecia'
			kod = bytes(message.encode('utf-8'))
			print(message)
			self.conn.sendall(kod)

	def crop_image(self, frame, wspolrzedne):
		try:
			frame = frame[wspolrzedne[0]:wspolrzedne[1],wspolrzedne[2]:wspolrzedne[3]]
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
			return frame
		except TypeError:
			message = 'error' + '#' + 'blad crop_image'
			kod = bytes(message.encode('utf-8'))
			print(message)
			self.conn.sendall(kod)

	def odczytaj_datamatrix(self, frame):
		kod_kreskowy = self.data_matrix(frame)
		kod_status = kod_kreskowy.decode('utf-8')
		kod_status = kod_status.split('#')[0]
		if kod_status == 'error':
			try:
				new_img_size = 200
				img_resized = cv2.resize(frame.copy(), (new_img_size, new_img_size))
				kod_pylib = decode(img_resized)
				kod_pylib = kod_pylib[0].data
				if kod_pylib:
					kod_kreskowy = kod_pylib
			except:
				message = 'error' + '#' + 'Wyjatek konwersji pylibdmtx'
				print(message)
				kod_kreskowy = bytes(message.encode('utf-8'))
		kod_status = kod_kreskowy.decode('utf-8')
		kod_status = kod_status.split('#')[0]
		if kod_status == 'error':
			self.zapisz_zdjecie(frame, '/home/pi/temp')																					
			try:
				kod_zxing = self.zxing_reader.decode('/home/pi/temp.jpg')
				if kod_zxing.raw is not None:
					kod_zxing = kod_zxing.raw
					print('zxing', kod_zxing)
					kod_kreskowy = bytes(kod_zxing.encode('utf-8'))
			except:
				message = 'error' + '#' + 'Wyjatek konwersji zxing'
				print(message)
				kod_kreskowy = bytes(message.encode('utf-8'))	
		return kod_kreskowy

	def _decode_coordinates(self, data):
		try:
			wspolrzedne = []
			wspolrzedne.append(int(data.split('#')[1]))
			wspolrzedne.append(int(data.split('#')[2]))
			wspolrzedne.append(int(data.split('#')[3]))
			wspolrzedne.append(int(data.split('#')[4]))
			return wspolrzedne
		except IndexError:
			message = 'error' + '#' + 'blad indeksowania'
			print(message)
			kod = bytes(message.encode('utf-8'))
			self.conn.sendall(kod)

	def odbierz_dane(self):
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			socketserver.TCPServer.allow_reuse_address = True
			s.bind((self.HOST, self.PORT))
			s.listen()
			while True:	 
				self.conn, addr = s.accept()
				try:
					with self.conn:
						print('Connected by', addr)
						while True:
							data = self.conn.recv(4096)
							if not data:
								break
							data = data.decode('utf-8')
							rodzaj_kodu = data.split('#')[0]

							if rodzaj_kodu == 'kod_1D':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:						
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:							
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.kod_kreskowy(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'kod_1D_1':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(1)
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:								
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:							
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.kod_kreskowy(frame)
											self.conn.sendall(kod_kreskowy) 
							elif rodzaj_kodu == 'kod_1D_2':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(2)
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:									
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:							
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.kod_kreskowy(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'kod_1D_3':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(3)
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:								
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:								
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.kod_kreskowy(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'kod_1D_4':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(4)
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:									
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:							
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.kod_kreskowy(frame)
											self.conn.sendall(kod_kreskowy)	
							elif rodzaj_kodu == 'kod_data_matrix_1':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(1)						
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:									
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:						
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.odczytaj_datamatrix(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'kod_data_matrix_2':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(2)						
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:								
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:								
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.odczytaj_datamatrix(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'kod_data_matrix_3':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(3)						
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:									
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:								
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.odczytaj_datamatrix(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'kod_data_matrix_4':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									self.zalacz_kamere(4)						
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:							
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:							
											self.zapisz_NAS(frame)							
											kod_kreskowy = self.odczytaj_datamatrix(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'kod_data_matrix':
								wspolrzedne =  self._decode_coordinates(data)
								if isinstance(wspolrzedne, list):
									frame = self.zrob_zdjecie()
									if type(frame) is np.ndarray:		
										frame = self.crop_image(frame,wspolrzedne)
										if type(frame) is np.ndarray:
											self.zapisz_NAS(frame)
											kod_kreskowy = self.odczytaj_datamatrix(frame)
											self.conn.sendall(kod_kreskowy)
							elif rodzaj_kodu == 'zapisz_zdjecie':
								frame = self.zrob_zdjecie()
								if type(frame) is np.ndarray:
									self.zapisz_zdjecie(frame, 'temp')	
							elif rodzaj_kodu == 'zwroc_ramke':
								frame = self.zrob_zdjecie()
								if type(frame) is np.ndarray:
									a = pickle.dumps(frame)
									message = struct.pack("Q", len(a)) + a
									self.conn.sendall(message)
							elif rodzaj_kodu == 'wylacz_kamery':
								self.zalacz_kamere(0)
							else:
								message = 'error' + '#' + 'bledna komenda (kod_1D lub kod_data_matrix)'								
								self.conn.sendall(message.encode('utf-8'))
				except socket.error as e:
					print("self.connection error: %s" % e)
				finally:
					self.conn.close()

	def main(self):
		try:
			self.odbierz_dane()
		except OSError as error:
			print(error)
		finally:
			GPIO.cleanup(self.Selection)
			GPIO.cleanup(self.Enable1)
			GPIO.cleanup(self.Enable2)
	
	def __exit__(self):
		self.camera.release()

######################################## program glowny
if __name__ == "__main__":
	barcode = Barcode()
	barcode.main()
