#include <iostream>
#include <string>

#include "mnist/mnist_reader.hpp"
#include "bi_transformer.hpp"

using bi_transformer::BiTransformer;
using bi_transformer::ConjuntoImagenes;
using bi_transformer::Etiquetas;
using bi_transformer::normalizar_imagenes;
using bi_transformer::parsear_tamano;

int main(int argc, char* argv[]) {
	const std::string ruta_datos = argc > 1 ? argv[1] : "mnist-dataset";
	const std::size_t limite_entrenamiento = argc > 2 ? parsear_tamano(argv[2], 8000) : 8000;
	const std::size_t limite_prueba = argc > 3 ? parsear_tamano(argv[3], 2000) : 2000;
	const std::size_t filas_mapa = argc > 4 ? parsear_tamano(argv[4], 8) : 8;
	const std::size_t columnas_mapa = argc > 5 ? parsear_tamano(argv[5], 8) : 8;
	const std::size_t epocas_mapa = argc > 6 ? parsear_tamano(argv[6], 8) : 8;

	std::cout << "Cargando MNIST desde: " << ruta_datos << '\n';
	std::cout << "Limite entrenamiento: " << limite_entrenamiento << '\n';
	std::cout << "Limite prueba: " << limite_prueba << '\n';

	auto dataset = mnist::read_dataset<std::vector, std::vector, double, std::uint8_t>(ruta_datos, limite_entrenamiento, limite_prueba);

	normalizar_imagenes(dataset.training_images);
	normalizar_imagenes(dataset.test_images);

	if (dataset.training_images.empty() || dataset.test_images.empty()) {
		std::cerr << "No se pudieron cargar los datos MNIST. Verifica la ruta y los archivos del dataset." << '\n';
		return 1;
	}

	std::cout << "Entrenando BiTransformer..." << '\n';
	BiTransformer bi_transformer(filas_mapa, columnas_mapa);
	bi_transformer.entrenar(dataset.training_images, dataset.training_labels, epocas_mapa);

	std::cout << "Evaluando..." << '\n';
	const double exactitud = bi_transformer.evaluar(dataset.test_images, dataset.test_labels);

	std::cout << "Exactitud: " << exactitud * 100.0 << "%" << '\n';

	const std::size_t ejemplos = std::min<std::size_t>(10, std::min(dataset.test_images.size(), dataset.test_labels.size()));
	for (std::size_t i = 0; i < ejemplos; ++i) {
		const int prediccion = bi_transformer.predecir(dataset.test_images[i]);
		std::cout << "Muestra " << i + 1 << ": real=" << static_cast<int>(dataset.test_labels[i])
				  << ", predicha=" << prediccion << '\n';
	}

	return 0;
}
