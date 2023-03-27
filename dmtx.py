#!/usr/bin/env python3

#Python3 Class for Datamatrix decoding

#Copyright 2022 JM Elektronik Ltd.

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

################## v 0.3
import cv2
import numpy as np
import json
import galois
import os

# white_line_threshold = 35000

class DMTX:
	def __init__(self, stride = 2, white_line_threshold = 39000, debug = False):
					
		'''
		Inicjalizuję klasę.
		'''
		self.PIXEL_NUMBER_BORDER_DETECT = 5 				# liczba pikseli od brzegu do uśrednienia przy detekcji ramki
		self.SOLID_DOTTED_THRESHOLD = 80
		self.THRESHOLD = [71]						# lista wartosci do sprawdzenia (musza byc nieparzyste)
		self.CONTRASTS = [1.5]								# lista wartosci (0 - 4)
		self.DEBUG = debug									# tryb diagnostyczny
		self.RECTANGLE_SIZE = [210]			  				# okresla wielkosc pola w ktorym szukany jest dmtx (musi byc lista) stara metoda
		self.STRIDE = stride								# co ile px przesuwa sie szukajacy kwadrat
		self.window_delay_ms = 0 							# czas wyswietlania okien diagnostycznych 0 - czeka na klawisz
		self.WHITE_LINE_THRESHOLD = white_line_threshold 	# suma linii powyzej ktorej usuwa linie
		self.dmtx_pixel_count = 0							# rozmiar kodu dmtx z ramka
		self.BLACK_WHITE_THRESHOLD = 122					# prog rozrozniania px na obrazie
		self.PIXEL_OFFSET = -3								# o tyle zmniejsza rozmiar px dmtx 
		self.DOT_FULL_THRESHOLD = 1100						# podzial miedzy linia pelna a kropkowana
		home_path = '/home/pi'
		self.DMTX_LIB_DIR = home_path + '/' + 'dmtx_dictionaries'	
		self.DMTX_DICTIONARY_PATH = self.DMTX_LIB_DIR + '/' + 'dictionary.json'	
		self.dmtx_dictionary = {} 							# slownik dmtx
		self.GF = galois.Field(2**8, irreducible_poly="x^8 + x^5 + x^3 + x^2 + 1")
		# self.RS = galois.ReedSolomon(255, d = 15, field=GF)	
		self.RS = None
															# d - długość wielomianu korekcji błędu:
															# 6, 8, 11, 12, 13, 15, 19, 21, 25, 29, 37, 43, 49, 57, 63, 69
		self._load_all_patterns()
		self._load_dmtx_dictionary()

	def _load_all_patterns(self):
		'''
		Laduje slowniki z L -kami dla poszczegolnych rozmiarow
		'''
		self.x10 = self._load_dmtx_pattern('10x10')
		self.x12 = self._load_dmtx_pattern('12x12')
		self.x14 = self._load_dmtx_pattern('14x14')
		self.x16 = self._load_dmtx_pattern('16x16')
		self.x18 = self._load_dmtx_pattern('18x18')
		self.x20 = self._load_dmtx_pattern('20x20')
		self.x22 = self._load_dmtx_pattern('22x22')
		self.x24 = self._load_dmtx_pattern('24x24')
		self.x26 = self._load_dmtx_pattern('26x26')

	def _load_dmtx_pattern(self, pattern_name):
		'''
		Laduje plik ze wspolrzednymi bajtow (L bracket) dla danego rozmiaru dmtx
		'''
		path = self.DMTX_LIB_DIR + '/' + pattern_name + '.json'
		with open(path, 'r') as file:
			pattern = json.load(file)
		return pattern

	def _load_dmtx_dictionary(self):
		'''
		Laduje plik ze slownikiem do konwersji bajtow odczytanych na dane
		'''
		with open (self.DMTX_DICTIONARY_PATH, 'r') as file:
			self.dmtx_dictionary = json.load(file)

	def dynamic_threshold(self, image):
		'''
		Szuka najlepszego threshold -u z listy i konwertuje
		'''
		img = image.copy()
		img_width = int(img.shape[1])
		img_height = int(img.shape[0])
		wyniki = []
		threshold = []
		for thr in self.THRESHOLD:
			thr_img = cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thr, 5)
			wyniki.append(np.mean(thr_img))
			threshold.append(thr)

		thr = threshold[wyniki.index(min(wyniki))]
		thr_img = cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thr, 5)
		if self.DEBUG == True:
			cv2.imshow('Thresholded image', thr_img)
			cv2.waitKey(self.window_delay_ms)
			cv2.destroyWindow('Thresholded image')
		return thr_img

	def dynamic_contast(self, image, kernel_size = 1):
		'''
		Szuka najlepszego kontrastu z listy i zwraca
		'''
		img = image.copy()
		wyniki = []
		contr = []
		threshold = []
		ksize= (kernel_size, kernel_size)
		img = cv2.blur(img,ksize)

		for contast in self.CONTRASTS:
			img_cont = cv2.convertScaleAbs(img.copy(), alpha=contast)
			wyniki.append(np.std(img_cont))
			contr.append(contast)
			threshold.append(contast)

		cont = threshold[wyniki.index(min(wyniki))]
		img_contrasted = cv2.convertScaleAbs(img.copy(), alpha=cont)
		if self.DEBUG == True:
			cv2.imshow('Contrasted image', img_contrasted)
			cv2.waitKey(self.window_delay_ms)	
			cv2.destroyWindow('Contrasted image')	
		return img_contrasted

	def resize(self, image, div_factor):
		'''
		Dzieli rozmiar przez div_factor
		'''
		img = image.copy()
		size = div_factor
		width = int(img.shape[1] / size)
		height = int(img.shape[0] / size)
		dim = (width, height)
		resized_img = cv2.resize(img, dim)
		if self.DEBUG == True:
			cv2.imshow('Resized image', resized_img)
			cv2.waitKey(self.window_delay_ms)	
			cv2.destroyWindow('Resized image')	
		return resized_img

	def add_white_border(self, image, border_width = 5):
		'''
		Dodaje biala ramke na okolo, na wypadek gdyby kod dotykal krawedzi
		'''
		img = image.copy()
		image = np.pad(image, pad_width = border_width, mode = 'constant', constant_values = 255)
		if self.DEBUG == True:
			cv2.imshow('White border image', image)
			cv2.waitKey(self.window_delay_ms)
			cv2.destroyWindow('White border image')
		return image	

	def remove_background(self, image):
		'''
		Usuniecie niepotrzebnego obszaru wokół kodu DM
		'''
		# Parametry użyte w funkcji są dobrane dla obrazu o rozmiarze 300x300 px
		# Użycie obrazka innego rozmiaru spowoduje, że funkcja nie będzie działać poprawnie
		# Dlatego obraz na początku jest skalowany do rozmiaru 300x300
		# Na koniec współrzędne wierzchołków prostokąta zawierającego kod są skalowane do 
		# współrzędnych oryginalnego obrazka i kod wycinany jest z obrazka przesłanego do funkcji


		# kinga
		# To skalowanie chyba jednak nie bedzie potrzebne, jeżeli na koncu
		# i tak zmieniamy rozmiar na 210x210
		try:		
			new_img_size = 300
			img_original = image.copy()
			img_original_size = img_original.shape
			img_original_height = img_original_size[0]
			img_original_width = img_original_size[1]
			height_ratio = img_original_height / new_img_size
			width_ratio = img_original_width / new_img_size
			
			img_resized = cv2.resize(img_original.copy(), (new_img_size, new_img_size))
			harris = cv2.cornerHarris(img_resized, 21, 21, 0.01)
			if self.DEBUG == True:
				cv2.imshow('harris', harris)
				cv2.waitKey(self.window_delay_ms)
				cv2.destroyWindow('harris')

			_, thresh = cv2.threshold(harris, 0.03 * harris.max(), 255, cv2.THRESH_BINARY)
			thresh = thresh.astype('uint8')
			if self.DEBUG == True:
				cv2.imshow('thresh', thresh)
				cv2.waitKey(self.window_delay_ms)
				cv2.destroyWindow('thresh')

			contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

			areas = [cv2.contourArea(cv2.convexHull(x)) for x in contours]
			max_area_contour_index = areas.index(max(areas))
			rect = cv2.minAreaRect(contours[max_area_contour_index])
			

			if self.DEBUG == True:
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				e = cv2.drawContours(img_resized,[box],0,1,1)
				cv2.imshow('Kontury Kinga', e)
				cv2.waitKey(self.window_delay_ms)
				cv2.destroyWindow('Kontury Kinga')

			x,y,w,h = cv2.boundingRect(contours[max_area_contour_index])

			x_min = round(x * width_ratio)
			x_max = round((x + w) * width_ratio)
			y_min = round(y * height_ratio)
			y_max = round((y + h) * height_ratio)

			if x_min < 0:
				x = 0
			if x_max > img_original_width:
				x_max = img_original_width - 1
			if y_min < 0:
				y = 0
			if y_max > img_original_height:
				y_max = img_original_height - 1

			img = img_original[y_min:y_max, x_min:x_max]
			# z jakiegoś powodu z tym działa lepiej
			img = cv2.resize(img, (img_original_height, img_original_width))

			new_img_size = self.RECTANGLE_SIZE[0]
			img_resized = cv2.resize(img, (new_img_size, new_img_size))
			threshold_param = 69
			img = cv2.adaptiveThreshold(img_resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_param, 0)

			if self.DEBUG == True:
				cv2.imshow('Remove background', img)
				cv2.waitKey(self.window_delay_ms)
				cv2.destroyWindow('Remove background')
		except:
			message = 'error' + '#' + 'Wyjatek usuwania tla'
			print(message)
			img = bytes(message.encode('utf-8'))
			rect = [0,0,90.]
		return img, rect

	def find_contours(self, image):
		'''
		Znajduje drugi najwiekszy kontur, zwraca prostokat tego konturu
		'''
		img = image.copy()
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = cv2.drawContours(np.full_like(img ,255), contours, -1, 0, 1)
		if self.DEBUG == True:
			cv2.imshow('Contours', c)
			key = cv2.waitKey(self.window_delay_ms) & 0xFF 
			cv2.destroyWindow('Contours')
		areas = [cv2.contourArea(cv2.convexHull(x)) for x in contours]
		max_i = areas.index(max(areas))	

		second_max_i = areas.index(np.unique(areas)[-2])
		third_max_i = areas.index(np.unique(areas)[-3])
		if self.DEBUG == True:
			print('Kontury', areas)
			print('Index 2nd', second_max_i, 'Index 3rd', third_max_i)	
		max_contour = cv2.drawContours(np.full_like(img.copy(),255), [contours[second_max_i],contours[third_max_i]], -1, 0, 1)
		rect = cv2.minAreaRect(contours[second_max_i])
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		if self.DEBUG == True:
			print('Wspolrzedne rogow',box)
			print('Wspolrzedne (rozmiar, koordynaty, kat)',rect)
		dmtx_box = cv2.drawContours(img.copy(),[box],0,0,1)
		if self.DEBUG == True:	
			cv2.imshow('dmtx box', dmtx_box)
			key = cv2.waitKey(self.window_delay_ms) & 0xFF
			cv2.destroyWindow('dmtx box') 
		return rect

	def rotate_image(self, image, ang):
		'''
		Obraca o kat 
		'''
		try:
			img = image.copy()
			height, width = img.shape[:2]
			center = (width/2, height/2)
			rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=ang, scale=1)
			rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height), borderMode=cv2.BORDER_CONSTANT, borderValue = 255)
			if self.DEBUG == True:	
				cv2.imshow('Rotated image', rotated_image)
				key = cv2.waitKey(self.window_delay_ms) & 0xFF 
				cv2.destroyWindow('Rotated image')		
		except:
			message = 'error' + '#' + 'Wyjatek obrotu obrazu'
			print(message)
			rotated_image = bytes(message.encode('utf-8'))				
		return rotated_image

	def crop_rect(self, image, rect):
		img = image.copy()
		center = rect[0]
		size = rect[1]
		angle = rect[2]
		center, size = tuple(map(int, center)), tuple(map(int, size))
		height, width = img.shape[0], img.shape[1]
		M = cv2.getRotationMatrix2D(center, angle, 1)
		img_rot = cv2.warpAffine(img, M, (width, height))
		img_crop = cv2.getRectSubPix(img_rot, size, center)
		if self.DEBUG == True:	
			cv2.imshow('Cropped image', img_crop)
			key = cv2.waitKey(self.window_delay_ms) & 0xFF 
			cv2.destroyWindow('Cropped image')			
		return img_crop

	def remove_white_border(self, image):
		try:
			source_img = image.copy()
			ver_sum = np.sum(source_img, axis = 0)
			test_list = []
			index_list = []
			# Kasuje od pierwszej kolumny do line threshold
			test_list.append(['Deleting ver line'])
			for index, ver_line in enumerate(ver_sum):
				if ver_line >= self.WHITE_LINE_THRESHOLD:
					index_list.append(index)
					if self.DEBUG == True:
						test_list.append(ver_line)
				else:
					break
			for index in reversed(index_list):
				source_img = np.delete(source_img, (index), axis=1)
			index_list = []

			test_list.append(['Deleting hor line'])
			# Kasuje od pierwszego wiersza do line threshold
			hor_sum = np.sum(source_img, axis=1)
			for index, hor_line in enumerate(hor_sum):
				if hor_line >= self.WHITE_LINE_THRESHOLD:
					index_list.append(index)
					if self.DEBUG == True:	
						test_list.append(hor_line)
				else:
					break
			for index in reversed(index_list):
				source_img = np.delete(source_img, (index), axis=0)
			index_list = []

			test_list.append(['Deleting ver line'])
			# Kasuje od ostatniej kolumny do line threshold
			ver_sum = np.sum(source_img, axis = 0)
			max_index = len(ver_sum) - 1
			for index, ver_line in enumerate(reversed(ver_sum)):
				if ver_line >= self.WHITE_LINE_THRESHOLD:
					index_list.append(max_index - index)
					if self.DEBUG == True:	
						test_list.append(ver_line)
				else:
					break
			for index in index_list:
				source_img = np.delete(source_img, (index), axis=1)
			index_list = []

			test_list.append(['Deleting hor line'])
			# Kasuje od ostatniego wiersza do line threshold
			hor_sum = np.sum(source_img, axis=1)
			max_index = len(hor_sum) - 1
			for index, hor_line in enumerate(reversed(hor_sum)):
				if hor_line >= self.WHITE_LINE_THRESHOLD:
					index_list.append(max_index - index)
					if self.DEBUG == True:	
						test_list.append(hor_line)
				else:
					break
			for index in index_list:
				source_img = np.delete(source_img, (index), axis=0)
			index_list = []

			if self.DEBUG == True:	
				print(test_list)
				cv2.imshow('No white border image', source_img)
				key = cv2.waitKey(self.window_delay_ms) & 0xFF 
				cv2.destroyWindow('No white border image')	
		except:
			message = 'error' + '#' + 'Wyjatek usuwania obrysu'
			print(message)
			source_img = bytes(message.encode('utf-8'))		
		return source_img

	def matrix_swell(self, matrix, swell_factor = 5):
		'''
		Mnoży kazdy piksel matrycy razy swell factor
		'''
		try:
			matrix_rows = np.repeat(matrix, swell_factor, axis=1)
			matrix_rows_col = np.repeat(matrix_rows, swell_factor, axis=0)
			if self.DEBUG == True:	
				cv2.imshow('Swolled matrix', matrix_rows_col)
				key = cv2.waitKey(self.window_delay_ms) & 0xFF 	
				cv2.destroyWindow('Swolled matrix')	
			return matrix_rows_col
		except:
			message = 'error' + '#' + 'Wyjatek matrix_swell'
			print(message)
			return bytes(message.encode('utf-8'))		

	def regenarate(self, image):
		try:
			img = image.copy()
			dmtx_regenerated_size = (self.dmtx_pixel_count, self.dmtx_pixel_count)

			dmtx_regenerated = np.zeros(dmtx_regenerated_size, dtype = np.uint8)
			dmtx_pixel_height = img.shape[0] / self.dmtx_pixel_count
			dmtx_pixel_width = img.shape[1] / self.dmtx_pixel_count

			for ver_pixel in range(self.dmtx_pixel_count):
				for hor_pixel in range(self.dmtx_pixel_count):
					pixel_bl = [int((hor_pixel * dmtx_pixel_width) - self.PIXEL_OFFSET), int(((ver_pixel + 1) * dmtx_pixel_height)) + self.PIXEL_OFFSET]
					pixel_tl = [int((hor_pixel * dmtx_pixel_width) - self.PIXEL_OFFSET),int((ver_pixel * dmtx_pixel_height)) - self.PIXEL_OFFSET]
					#pixel_tr = [int(((hor_pixel + 1) * dmtx_pixel_width) + self.PIXEL_OFFSET),int((ver_pixel * dmtx_pixel_height)) - self.PIXEL_OFFSET]
					pixel_br = [int(((hor_pixel + 1) * dmtx_pixel_width) + self.PIXEL_OFFSET),int(((ver_pixel + 1) * dmtx_pixel_height)) + self.PIXEL_OFFSET]
					temp_img = img.copy()
					temp_img = temp_img[pixel_tl[1]:pixel_bl[1],pixel_tl[0]:pixel_br[0]]

					avg_col = cv2.mean(temp_img)[0]
					if avg_col <= self.BLACK_WHITE_THRESHOLD:
						regenerated_pixel = 0
					else:
						regenerated_pixel = 255

					dmtx_regenerated[ver_pixel][hor_pixel] = regenerated_pixel 

			if self.DEBUG == True:	
				cv2.imshow('Dmtx regenerated', dmtx_regenerated)
				key = cv2.waitKey(self.window_delay_ms) & 0xFF 
				cv2.destroyWindow('Dmtx regenerated')
		except:
			message = 'error' + '#' + 'Wyjatek regeneracji'
			print(message)
			dmtx_regenerated = bytes(message.encode('utf-8'))				
		return dmtx_regenerated

	def count_black_pixels(self, row, column, dmtx_pixel_width, dmtx_pixel_height, img, obrazek, offset):
		counter_black = 0
		row_start = round(row * dmtx_pixel_height) + offset
		row_stop = round((row + 1) * dmtx_pixel_height ) - offset
		if row_stop >= img.shape[0]: 
			row_stop = img.shape[0] - 1
		column_start = round(column * dmtx_pixel_width) + offset
		column_stop = round((column + 1) * dmtx_pixel_width ) - offset
		if column_stop >= img.shape[1]: 
			column_stop = img.shape[1] - 1

		print('pixels row:', row_stop - row_start)
		print('pixels column:', column_stop - column_start)

		for r in range(row_start, row_stop):
			for c in range(column_start, column_stop):
				if img[r][c] == 0:
					counter_black += 1

		for i in range(0, obrazek.shape[0]):
			obrazek[i][column_stop] = 123
		for i in range(0, obrazek.shape[1]):
			obrazek[row_stop][i] = 123

		# print('counter:{}, wielkośc kropy:{}'.format(counter_black, (row_stop - row_start) * (column_stop - column_start)))

		return counter_black

	def read_image_grayscale(self, path):	
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		if self.DEBUG == True:	
			cv2.imshow('Grayscale image', img)
			key = cv2.waitKey(self.window_delay_ms) & 0xFF
			cv2.destroyWindow('Grayscale image') 
		return img

	def grayscale_bin(self, image):
		try:
			img = image.copy()
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			(_, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			if self.DEBUG == True:
				cv2.imshow('Thresholded img', im_bw)
				key = cv2.waitKey(self.window_delay_ms) & 0xFF
				cv2.destroyWindow('Thresholded img')
		except:
			message = 'error' + '#' + 'Wyjatek konwersji na szary'
			print(message)
			im_bw = bytes(message.encode('utf-8'))		
		return im_bw

	def detect_dmtx_size(self, image):
		'''
		Sprawdza ilosc zmian wartosci na ramce i zwraca liste
		'''
		try:
			lista = []
			img = image.copy()
			thr_img = self.dynamic_threshold(img)
			first_x_rows = thr_img[self.PIXEL_NUMBER_BORDER_DETECT-1:self.PIXEL_NUMBER_BORDER_DETECT,:]
			last_x_rows = thr_img[-self.PIXEL_NUMBER_BORDER_DETECT:-self.PIXEL_NUMBER_BORDER_DETECT+1,:]
			first_x_columns = thr_img[:,self.PIXEL_NUMBER_BORDER_DETECT-1:self.PIXEL_NUMBER_BORDER_DETECT]
			last_x_columns = thr_img[:,-self.PIXEL_NUMBER_BORDER_DETECT:-self.PIXEL_NUMBER_BORDER_DETECT+1]

			first_x_rows_transitions = (np.diff(first_x_rows)!=0).sum()		
			last_x_rows_transitions = (np.diff(last_x_rows)!=0).sum()
			first_x_columns_transitions = (np.diff(first_x_columns,axis=0)!=0).sum()
			last_x_columns_transitions = (np.diff(last_x_columns,axis=0)!=0).sum()
			
			lista.append(first_x_rows_transitions)
			lista.append(last_x_rows_transitions)
			lista.append(first_x_columns_transitions)
			lista.append(last_x_columns_transitions)
			if self.DEBUG == True:
				print('Liczba przejsc ',org_list)
			lista = [v for i, v in enumerate(lista) if v % 2 == 0]
			lista = [v for i, v in enumerate(lista) if v >= 10]
			lista = [*set(lista)]									#usuwa powtorzenia
			if not lista:
				lista = org_list
				lista = [v for i, v in enumerate(lista) if v >= 9]
				lista = [*set(lista)]									#usuwa powtorzenia
				for index, value in enumerate(lista):
					lista[index] = value+1
			
			if self.DEBUG == True:
				print('Liczba przejsc po obrobce', lista)	
		except:
			message = 'error' + '#' + 'Wyjatek wielkosci kodu'
			print(message)
			lista = bytes(message.encode('utf-8'))							
		return lista

	def rotate_dmtx(self, image):
		try:
			img = image.copy()
			column = np.hsplit(img,self.dmtx_pixel_count)
			row = np.vsplit(img,self.dmtx_pixel_count)
			first_column = column[0]
			last_column = column[-1]
			first_column = first_column.reshape(1,self.dmtx_pixel_count)
			first_column = first_column[0]
			last_column = last_column.reshape(1,self.dmtx_pixel_count)
			last_column = last_column[0]
			first_row = row[0][0]
			last_row = row[-1][0]

			if sum(first_column) >= self.DOT_FULL_THRESHOLD and sum(last_column) < self.DOT_FULL_THRESHOLD and sum(first_row) < self.DOT_FULL_THRESHOLD and sum(last_row) >= self.DOT_FULL_THRESHOLD:
				img = np.rot90(img,k=2)
				if self.DEBUG == True:				
					cv2.imshow('Dmtx rotated 180', img)
					key = cv2.waitKey(self.window_delay_ms) & 0xFF 	
					cv2.destroyWindow('Dmtx rotated 180')		
				return img
			elif sum(first_column) < self.DOT_FULL_THRESHOLD and sum(last_column) >= self.DOT_FULL_THRESHOLD and sum(first_row) < self.DOT_FULL_THRESHOLD and sum(last_row) >= self.DOT_FULL_THRESHOLD:
				img = np.rot90(img,k=1)
				if self.DEBUG == True:				
					cv2.imshow('Dmtx rotated 90', img)
					key = cv2.waitKey(self.window_delay_ms) & 0xFF 
					cv2.destroyWindow('Dmtx rotated 90')					
				return img
			elif sum(first_column) < self.DOT_FULL_THRESHOLD and sum(last_column) >= self.DOT_FULL_THRESHOLD and sum(first_row) >= self.DOT_FULL_THRESHOLD and sum(last_row) < self.DOT_FULL_THRESHOLD:
				if self.DEBUG == True:				
					cv2.imshow('Dmtx rotated 0', img)
					key = cv2.waitKey(self.window_delay_ms) & 0xFF 	
					cv2.destroyWindow('Dmtx rotated 0')				
				return img
			elif sum(first_column) >= self.DOT_FULL_THRESHOLD and sum(last_column) < self.DOT_FULL_THRESHOLD and sum(first_row) >= self.DOT_FULL_THRESHOLD and sum(last_row) < self.DOT_FULL_THRESHOLD:
				img = np.rot90(img,k=3)
				if self.DEBUG == True:				
					cv2.imshow('Dmtx rotated 270', img)
					key = cv2.waitKey(self.window_delay_ms) & 0xFF 
					cv2.destroyWindow('Dmtx rotated 270')				
				return img
			else:
				col_row_sum = [sum(first_column), sum(last_column), sum(first_row), sum(last_row)]
				print('Blad porownania kolumn z threshold', col_row_sum)
				return bytes(col_row_sum.encode('utf-8'))
		except:
			message = 'error' + '#' + 'Wyjatek rotate_dmtx'
			print(message)
			return bytes(message.encode('utf-8'))

	def remove_border(self, img):
		img = np.delete(img, 0, axis = 0)	# first row
		img = np.delete(img, -1, axis = 0)	# last row
		img = np.delete(img, 0, axis = 1)	# first column
		img = np.delete(img, -1, axis = 1)	# last column
		if self.DEBUG == True:				
			cv2.imshow('Borderless img', img)
			key = cv2.waitKey(self.window_delay_ms) & 0xFF 	
		return img	
	
	def _convert_bw2bin(self, pixel):
		'''
		Konwertuje pixel obrazu na bit
		pixel: 0 - czarny, 255 - bialy
		return: 1 - czarny, 0 - bialy
		'''
		try:
			pixel = int(pixel)
			if pixel == 0:
				dmtx_bit = 1
			elif pixel == 255:
				dmtx_bit = 0
			else:
				print('Bledna wartosc pixela ', pixel)
			return dmtx_bit
		except :
			print('Blad konwersji bw2bin')
			return None

	def _convert_bajt_char(self, bajt):
		'''
		Konwertuje bajt danych na wartosc ze slownika
		bajt - int
		return - str
		'''
		try:
			return self.dmtx_dictionary[str(bajt)]['character']
		except:
			# kinga
			# print('Blad konwersji bajtu na znak', bajt)
			return('0')

	def remove_dmtx_outline(self, img):
		try:
			img = np.delete(img, 0, axis = 0)	# first row
			img = np.delete(img, -1, axis = 0)	# last row
			img = np.delete(img, 0, axis = 1)	# first column
			img = np.delete(img, -1, axis = 1)	# last column
			if self.DEBUG == True:				
				cv2.imshow('No outline dmtx', img)
				key = cv2.waitKey(self.window_delay_ms) & 0xFF 	
				cv2.destroyWindow('No outline dmtx')
			return img
		except:
			message = 'error' + '#' + 'Wyjatek remove_dmtx_outline'
			print(message)
			return bytes(message.encode('utf-8'))		

	def dmtx_to_codewords(self, dmtx_array):
		try:
			if self.dmtx_pixel_count == 10:
				L_list = [self.x10['L1'],self.x10['L2'],self.x10['L3'],self.x10['L4'],self.x10['L5'],
						self.x10['L6'],self.x10['L7'],self.x10['L8']]
			elif self.dmtx_pixel_count == 12:
				L_list = [self.x12['L1'],self.x12['L2'],self.x12['L3'],self.x12['L4'],self.x12['L5'],
						self.x12['L6'],self.x12['L7'],self.x12['L8'],self.x12['L9'],self.x12['L10'],
						self.x12['L11'],self.x12['L12']]
			elif self.dmtx_pixel_count ==14:
				L_list = [self.x14['L1'],self.x14['L2'],self.x14['L3'],self.x14['L4'],self.x14['L5'],
						self.x14['L6'],self.x14['L7'],self.x14['L8'],self.x14['L9'],self.x14['L10'],
						self.x14['L11'],self.x14['L12'],self.x14['L13'],self.x14['L14'],self.x14['L15'],
						self.x14['L16'],self.x14['L17'],self.x14['L18']]
			elif self.dmtx_pixel_count == 16:
				L_list = [self.x16['L1'],self.x16['L2'],self.x16['L3'],self.x16['L4'],self.x16['L5'],
						self.x16['L6'],self.x16['L7'],self.x16['L8'],self.x16['L9'],self.x16['L10'],
						self.x16['L11'],self.x16['L12'],self.x16['L13'],self.x16['L14'],self.x16['L15'],
						self.x16['L16'],self.x16['L17'],self.x16['L18'],self.x16['L19'],self.x16['L20'],
						self.x16['L21'],self.x16['L22'],self.x16['L23'],self.x16['L24']]
			elif self.dmtx_pixel_count == 18:
				L_list = [self.x18['L1'],self.x18['L2'],self.x18['L3'],self.x18['L4'],self.x18['L5'],
						self.x18['L6'],self.x18['L7'],self.x18['L8'],self.x18['L9'],self.x18['L10'],
						self.x18['L11'],self.x18['L12'],self.x18['L13'],self.x18['L14'],self.x18['L15'],
						self.x18['L16'],self.x18['L17'],self.x18['L18'],self.x18['L19'],self.x18['L20'],
						self.x18['L21'],self.x18['L22'],self.x18['L23'],self.x18['L24'],self.x18['L25'],
						self.x18['L26'],self.x18['L27'],self.x18['L28'],self.x18['L29'],self.x18['L30'],
						self.x18['L31'],self.x18['L32']]
			elif self.dmtx_pixel_count == 20:
				L_list = [self.x20['L1'],self.x20['L2'],self.x20['L3'],self.x20['L4'],self.x20['L5'],
						self.x20['L6'],self.x20['L7'],self.x20['L8'],self.x20['L9'],self.x20['L10'],
						self.x20['L11'],self.x20['L12'],self.x20['L13'],self.x20['L14'],self.x20['L15'],
						self.x20['L16'],self.x20['L17'],self.x20['L18'],self.x20['L19'],self.x20['L20'],
						self.x20['L21'],self.x20['L22'],self.x20['L23'],self.x20['L24'],self.x20['L25'],
						self.x20['L26'],self.x20['L27'],self.x20['L28'],self.x20['L29'],self.x20['L30'],
						self.x20['L31'],self.x20['L32'],self.x20['L33'],self.x20['L34'],self.x20['L35'],
						self.x20['L36'],self.x20['L37'],self.x20['L38'],self.x20['L39'],self.x20['L40']]
			elif self.dmtx_pixel_count == 22:
				L_list = [self.x22['L1'],self.x22['L2'],self.x22['L3'],self.x22['L4'],self.x22['L5'],
						self.x22['L6'],self.x22['L7'],self.x22['L8'],self.x22['L9'],self.x22['L10'],
						self.x22['L11'],self.x22['L12'],self.x22['L13'],self.x22['L14'],self.x22['L15'],
						self.x22['L16'],self.x22['L17'],self.x22['L18'],self.x22['L19'],self.x22['L20'],
						self.x22['L21'],self.x22['L22'],self.x22['L23'],self.x22['L24'],self.x22['L25'],
						self.x22['L26'],self.x22['L27'],self.x22['L28'],self.x22['L29'],self.x22['L30'],
						self.x22['L31'],self.x22['L32'],self.x22['L33'],self.x22['L34'],self.x22['L35'],
						self.x22['L36'],self.x22['L37'],self.x22['L38'],self.x22['L39'],self.x22['L40'],
						self.x22['L41'],self.x22['L42'],self.x22['L43'],self.x22['L44'],self.x22['L45'],
						self.x22['L46'],self.x22['L47'],self.x22['L48'],self.x22['L49'],self.x22['L50']]					
			elif self.dmtx_pixel_count == 24:
				L_list = [self.x24['L1'],self.x24['L2'],self.x24['L3'],self.x24['L4'],self.x24['L5'],
						self.x24['L6'],self.x24['L7'],self.x24['L8'],self.x24['L9'],self.x24['L10'],
						self.x24['L11'],self.x24['L12'],self.x24['L13'],self.x24['L14'],self.x24['L15'],
						self.x24['L16'],self.x24['L17'],self.x24['L18'],self.x24['L19'],self.x24['L20'],
						self.x24['L21'],self.x24['L22'],self.x24['L23'],self.x24['L24'],self.x24['L25'],
						self.x24['L26'],self.x24['L27'],self.x24['L28'],self.x24['L29'],self.x24['L30'],
						self.x24['L31'],self.x24['L32'],self.x24['L33'],self.x24['L34'],self.x24['L35'],
						self.x24['L36'],self.x24['L37'],self.x24['L38'],self.x24['L39'],self.x24['L40'],
						self.x24['L41'],self.x24['L42'],self.x24['L43'],self.x24['L44'],self.x24['L45'],
						self.x24['L46'],self.x24['L47'],self.x24['L48'],self.x24['L49'],self.x24['L50'],
						self.x24['L51'],self.x24['L52'],self.x24['L53'],self.x24['L54'],self.x24['L55'],
						self.x24['L56'],self.x24['L57'],self.x24['L58'],self.x24['L59'],self.x24['L60']]
			elif self.dmtx_pixel_count == 26:
				L_list = [self.x26['L1'],self.x26['L2'],self.x26['L3'],self.x26['L4'],self.x26['L5'],
						self.x26['L6'],self.x26['L7'],self.x26['L8'],self.x26['L9'],self.x26['L10'],
						self.x26['L11'],self.x26['L12'],self.x26['L13'],self.x26['L14'],self.x26['L15'],
						self.x26['L16'],self.x26['L17'],self.x26['L18'],self.x26['L19'],self.x26['L20'],
						self.x26['L21'],self.x26['L22'],self.x26['L23'],self.x26['L24'],self.x26['L25'],
						self.x26['L26'],self.x26['L27'],self.x26['L28'],self.x26['L29'],self.x26['L30'],
						self.x26['L31'],self.x26['L32'],self.x26['L33'],self.x26['L34'],self.x26['L35'],
						self.x26['L36'],self.x26['L37'],self.x26['L38'],self.x26['L39'],self.x26['L40'],
						self.x26['L41'],self.x26['L42'],self.x26['L43'],self.x26['L44'],self.x26['L45'],
						self.x26['L46'],self.x26['L47'],self.x26['L48'],self.x26['L49'],self.x26['L50'],
						self.x26['L51'],self.x26['L52'],self.x26['L53'],self.x26['L54'],self.x26['L55'],
						self.x26['L56'],self.x26['L57'],self.x26['L58'],self.x26['L59'],self.x26['L60'],
						self.x26['L61'],self.x26['L62'],self.x26['L63'],self.x26['L64'],self.x26['L65'],
						self.x26['L66'],self.x26['L67'],self.x26['L68'],self.x26['L69'],self.x26['L70'],
						self.x26['L71'],self.x26['L72']]					
			codeword_list = []
			test_list = []
			for pattern in L_list:
				codeword = ''
				for dmtx_pixel_coord in pattern:
					dmtx_pixel = dmtx_array[dmtx_pixel_coord[1]][dmtx_pixel_coord[0]]
					dmtx_pixel = self._convert_bw2bin(dmtx_pixel)
					codeword += str(dmtx_pixel)
				codeword = int(codeword, 2)
				codeword_list.append(codeword)
				if self.DEBUG == True:
					test_list.append(codeword)
			if self.DEBUG == True:
				print(test_list)
			return codeword_list
		except:
			message = 'error' + '#' + 'Wyjatek codewords'
			print(message)
			return bytes(message.encode('utf-8'))			

	def _set_RS_param(self):
		# 6, 8, 11, 12, 13, 15, 19, 21, 25, 29, 37, 43, 49, 57, 63, 69
		if self.dmtx_pixel_count == 10:
			d = 6
		elif self.dmtx_pixel_count == 12:
			d = 8
		elif self.dmtx_pixel_count == 14:
			d = 11
		elif self.dmtx_pixel_count == 16:
			d = 13
		elif self.dmtx_pixel_count == 18:
			d = 15
		elif self.dmtx_pixel_count == 20:
			d = 19
		elif self.dmtx_pixel_count == 22:
			d = 21
		elif self.dmtx_pixel_count == 24:
			d = 25
		elif self.dmtx_pixel_count == 26:
			d = 29
		else:
			print('Blad ilosci pixeli (dmtx_pixel_count)')
		self.RS = galois.ReedSolomon(255, d = d, field=self.GF)

	def encode_RS(self, dmtx_data):
		'''
		Wylicza kodowanie Reeda Solomona dla danych wejsciowych i kod korekcyjny

		'''
		self._set_RS_param()
		encoded_data = self.RS.encode(dmtx_data)
		if self.DEBUG == True:	
			print(encoded_data)
		return encoded_data

	def decode_RS(self, dmtx_data):
		'''
		Dekoduje Reeda Solomona i zwraca
		'''
		self._set_RS_param()
		decoded, errors = self.RS.decode(dmtx_data, errors=True)
		if self.DEBUG == True:	
			print('Zdekodowano',decoded,'Bledy',errors)
		return decoded, errors

	def convert_code(self, code):
		try:
			dmtx_code = ''
			end_data = False
			for c in code:
				if not end_data:
					dmtx_char = self._convert_bajt_char(c)
					if dmtx_char == 'PAD':
						end_data = True
					else:
						dmtx_code += dmtx_char
			return dmtx_code
		except:
			message = 'error' + '#' + 'Wyjatek convert_code'
			print(message)
			return message
	


############# OLD ##############
def remove_background_old(self, image, blur_factor = 11):
	'''
	Robi detekcję krawędzi i blur.
	Nastepnie szuka kwadratu o najwiekszej jasnosci
	'''
	img = image.copy()
	blurred = cv2.blur(img.copy(), (blur_factor, blur_factor))

	ddepth = cv2.CV_32F
	grad_x = cv2.Sobel(blurred.copy(), ddepth=ddepth, dx=1, dy=0, ksize=-1)
	grad_y = cv2.Sobel(blurred.copy(), ddepth=ddepth, dx=0, dy=1, ksize=-1)
	gradient = cv2.subtract(grad_x, grad_y)
	gradient = cv2.convertScaleAbs(gradient)

	if self.DEBUG == True:
		cv2.imshow('Sobel X Y using Sobel() function', gradient)
		cv2.waitKey(self.window_delay_ms)
		cv2.destroyWindow('Sobel X Y using Sobel() function')

	gradient_width = gradient.shape[1]
	gradient_height = gradient.shape[0]

	wyniki = []
	for rect_size in self.RECTANGLE_SIZE:
		gradient_searchable_width = gradient_width - rect_size
		gradient_searchable_height = gradient_height - rect_size

		for x in range(0, gradient_searchable_width, self.STRIDE):
			for y in range(0, gradient_searchable_height, self.STRIDE):
				temp_gradient = gradient.copy()
				temp_gradient = temp_gradient[x:x+rect_size,y:y+rect_size]
				suma = np.sum(temp_gradient)
				wyniki.append([rect_size,x,y,suma])
	#print(wyniki)
	max_val = np.amax(wyniki)
	i,j = np.where(wyniki == max_val)

	if len(i) > 1:
		i = i[-1]
		j = j[-1]

	wynik = wyniki[int(i)]
	if self.DEBUG == True:
		print('Rozm szukanego pola:',wynik[0], 'x:', wynik[1], 'y:',wynik[2], 'Suma pola:', wynik[3])
	x = wynik[1]
	y = wynik[2]
	gradient_width = wynik[1] + wynik[0]
	gradient_height = wynik[2] + wynik[0]		
	cropped_img = image[x:gradient_width,y:gradient_height]
	if self.DEBUG == True:
		cv2.imshow('Cropped image', cropped_img)
		cv2.waitKey(self.window_delay_ms)	
		cv2.destroyWindow('Cropped image')	
	# print('image width:', gradient_width)
	# print('image height:', gradient_height)
	return cropped_img		

def remove_white_border_old(self, image):
	source_img = image.copy()
	ver_sum = np.sum(source_img, axis = 0)
	test_list = []
	# Kasuje od pierwszej kolumny do line threshold
	test_list.append(['Deleting ver line'])
	for ver_line in ver_sum:
		if ver_line >= self.WHITE_LINE_THRESHOLD:
			index = list(ver_sum).index(ver_line)
			if self.DEBUG == True:	
				test_list.append(ver_line)
			source_img = np.delete(source_img,(index), axis=1)
		else:
			break
	test_list.append(['Deleting hor line'])
	# Kasuje od pierwszego wiersza do line threshold
	hor_sum = np.sum(source_img, axis=1)	
	for hor_line in hor_sum:
		if hor_line >= self.WHITE_LINE_THRESHOLD:
			index = list(hor_sum).index(hor_line)
			if self.DEBUG == True:	
				test_list.append(hor_line)
			source_img = np.delete(source_img,(index),axis=0)
		else:
			break
	test_list.append(['Deleting ver line'])		
	# Kasuje od ostatniej kolumny do line threshold
	ver_sum = np.sum(source_img, axis = 0)
	for ver_line in reversed(ver_sum):
		if ver_line >= self.WHITE_LINE_THRESHOLD:
			index = list(ver_sum).index(ver_line)
			if self.DEBUG == True:	
				test_list.append(ver_line)
			source_img = np.delete(source_img,(index), axis=1)
		else:
			break
	test_list.append(['Deleting hor line'])
	# Kasuje od ostatniego wiersza do line threshold
	hor_sum = np.sum(source_img, axis=1)	
	for hor_line in reversed(hor_sum):
		if hor_line >= self.WHITE_LINE_THRESHOLD:
			index = list(hor_sum).index(hor_line)
			if self.DEBUG == True:	
				test_list.append(hor_line)
			source_img = np.delete(source_img,(index),axis=0)
		else:
			break

	if self.DEBUG == True:	
		print(test_list)
		cv2.imshow('No white border image', source_img)
		key = cv2.waitKey(self.window_delay_ms) & 0xFF 
		cv2.destroyWindow('No white border image')	
	return source_img
