#ifndef BI_TRANSFORMER_HPP
#define BI_TRANSFORMER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "bi_transformer_utils.hpp"
#include "mapa_kohonen.hpp"
#include "red_hopfield.hpp"

namespace bi_transformer {

class BiTransformer {
public:
	BiTransformer(std::size_t filas_mapa, std::size_t columnas_mapa)
		: mapa_(filas_mapa, columnas_mapa, 28u * 28u), hopfield_(filas_mapa * columnas_mapa), entrenado_(false) {}

	void entrenar(const ConjuntoImagenes& imagenes, const Etiquetas& etiquetas, std::size_t epocas_mapa) {
		const std::size_t total = std::min(imagenes.size(), etiquetas.size());
		if (total == 0) {
			return;
		}

		ConjuntoImagenes imagenes_entrenamiento(imagenes.begin(), imagenes.begin() + static_cast<std::ptrdiff_t>(total));
		mapa_.entrenar(imagenes_entrenamiento, epocas_mapa);

		const std::size_t dimension_salida = mapa_.dimension_salida();
		std::vector<Imagen> acumulados(10, Imagen(dimension_salida, 0.0));
		std::vector<std::size_t> conteos(10, 0);

		for (std::size_t i = 0; i < total; ++i) {
			const int etiqueta = static_cast<int>(etiquetas[i]);
			if (etiqueta < 0 || etiqueta > 9) {
				continue;
			}

			const Imagen activaciones = mapa_.transformar(imagenes[i]);
			Imagen& acumulado = acumulados[static_cast<std::size_t>(etiqueta)];

			for (std::size_t j = 0; j < dimension_salida; ++j) {
				acumulado[j] += activaciones[j];
			}

			++conteos[static_cast<std::size_t>(etiqueta)];
		}

		prototipos_.clear();
		prototipos_continuos_.clear();
		etiquetas_prototipos_.clear();

		for (int etiqueta = 0; etiqueta <= 9; ++etiqueta) {
			const std::size_t cantidad = conteos[static_cast<std::size_t>(etiqueta)];
			if (cantidad == 0) {
				continue;
			}

			Imagen promedio_clase = acumulados[static_cast<std::size_t>(etiqueta)];
			for (double& valor : promedio_clase) {
				valor /= static_cast<double>(cantidad);
			}

			prototipos_continuos_.push_back(promedio_clase);
			prototipos_.push_back(convertir_a_bipolar(promedio_clase));
			etiquetas_prototipos_.push_back(static_cast<std::uint8_t>(etiqueta));
		}

		hopfield_.entrenar(prototipos_);
		entrenado_ = !prototipos_.empty();
	}

	int predecir(const Imagen& imagen) const {
		if (!entrenado_) {
			return -1;
		}

		const Imagen activaciones = mapa_.transformar(imagen);
		const Imagen estado = convertir_a_bipolar(activaciones);
		const Imagen recuperado = hopfield_.recuperar(estado);

		int mejor_etiqueta = -1;
		double mejor_distancia = std::numeric_limits<double>::max();

		for (std::size_t i = 0; i < prototipos_.size(); ++i) {
			const double distancia_continua = distancia_cuadrada(activaciones, prototipos_continuos_[i]);
			const double distancia_bipolar = distancia_cuadrada(recuperado, prototipos_[i]);
			const double distancia = distancia_continua + 0.15 * distancia_bipolar;
			if (distancia < mejor_distancia) {
				mejor_distancia = distancia;
				mejor_etiqueta = static_cast<int>(etiquetas_prototipos_[i]);
			}
		}

		return mejor_etiqueta;
	}

	double evaluar(const ConjuntoImagenes& imagenes, const Etiquetas& etiquetas) const {
		const std::size_t total = std::min(imagenes.size(), etiquetas.size());
		if (total == 0) {
			return 0.0;
		}

		std::size_t aciertos = 0;
		for (std::size_t i = 0; i < total; ++i) {
			const int prediccion = predecir(imagenes[i]);
			if (prediccion == static_cast<int>(etiquetas[i])) {
				++aciertos;
			}
		}

		return static_cast<double>(aciertos) / static_cast<double>(total);
	}

private:
	MapaKohonen mapa_;
	RedHopfield hopfield_;
	std::vector<Imagen> prototipos_continuos_;
	std::vector<Imagen> prototipos_;
	std::vector<std::uint8_t> etiquetas_prototipos_;
	bool entrenado_;
};

} // namespace bi_transformer

#endif