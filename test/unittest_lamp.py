import unittest
import numpy as np
import cv2
import os
from framework import lamp, ImageCS  # Предполагается, что lamp и ImageCS определены в вашем проекте

class TestLamp(unittest.TestCase):

    def setUp(self):
        # Создание тестового изображения и матрицы
        self.image_path = "test_image.png"
        self.image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        cv2.imwrite(self.image_path, self.image)
        self.matrix = np.random.randn(256, 256)  # Генерация случайной матрицы с правильными размерами
        self.M = 128
        self.K = 10

    def tearDown(self):
        # Удаление тестового изображения
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

    def test_lamp_output_type(self):
        # Проверка типа выходных данных функции lamp
        result = lamp(self.image_path, self.matrix, self.M, self.K)
        self.assertIsInstance(result, ImageCS)

    def test_lamp_output_shape(self):
        # Проверка, что восстановленное изображение имеет тот же размер, что и входное
        result = lamp(self.image_path, self.matrix, self.M, self.K)
        self.assertEqual(result.get_Image().shape, self.image.shape)

    def test_lamp_cr_psnr_values(self):
        # Проверка, что CR и PSNR вычислены и находятся в допустимых пределах
        result = lamp(self.image_path, self.matrix, self.M, self.K)
        cr = result.get_CR()
        psnr = result.get_PSNR()
        self.assertGreater(cr, 0.)  # Коэффициент сжатия должен быть положительным
        self.assertGreater(psnr, 0.)  # PSNR должен быть положительным

    def test_lamp_invalid_image_path(self):
        # Проверка, что неверный путь к изображению вызывает ошибку
        with self.assertRaises(cv2.error):
            lamp("invalid_path.png", self.matrix, self.M, self.K)

    def test_lamp_invalid_matrix_dimensions(self):
        # Проверка, что неверные размеры матрицы вызывают ошибку
        invalid_matrix = np.random.rand(128, 128)  # Неверные размеры
        with self.assertRaises(ValueError):
            lamp(self.image_path, invalid_matrix, self.M, self.K)

    def test_lamp_zero_measurements(self):
        # Проверка, что нулевое количество измерений (M=0) вызывает ошибку
        with self.assertRaises(ValueError):
            lamp(self.image_path, self.matrix, 0, self.K)

    def test_lamp_zero_sparsity(self):
        # Проверка, что нулевая разреженность (K=0) возвращает нулевую реконструкцию
        result = lamp(self.image_path, self.matrix, self.M, 0)
        reconstructed_image = result.get_Image()
        self.assertTrue(np.all(reconstructed_image == 0))

    def test_lamp_high_sparsity(self):
        # Проверка, что высокая разреженность (K > N) не вызывает ошибку
        result = lamp(self.image_path, self.matrix, self.M, 300)
        self.assertIsInstance(result, ImageCS)

if __name__ == "__main__":
    unittest.main()
