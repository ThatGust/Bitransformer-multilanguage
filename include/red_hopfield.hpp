#ifndef RED_HOPFIELD_HPP
#define RED_HOPFIELD_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "bi_transformer_utils.hpp"

namespace bi_transformer {

class RedHopfield {
public:
	explicit RedHopfield(std::size_t dimension) : dimension_(dimension), pesos_(dimension * dimension, 0.0) {}

	void entrenar(const std::vector<Imagen>& patrones) {
		std::fill(pesos_.begin(), pesos_.end(), 0.0);

		if (dimension_ == 0 || patrones.empty()) {
			return;
		}

		for (const auto& patron : patrones) {
			for (std::size_t i = 0; i < dimension_; ++i) {
				for (std::size_t j = i + 1; j < dimension_; ++j) {
					const double valor = patron[i] * patron[j];
					pesos_[i * dimension_ + j] += valor;
					pesos_[j * dimension_ + i] += valor;
				}
			}
		}

		const double escala = 1.0 / static_cast<double>(dimension_);
		for (double& peso : pesos_) {
			peso *= escala;
		}

		for (std::size_t i = 0; i < dimension_; ++i) {
			pesos_[i * dimension_ + i] = 0.0;
		}
	}

	Imagen recuperar(const Imagen& entrada, std::size_t iteraciones = 6) const {
		Imagen estado = entrada;

		for (std::size_t iteracion = 0; iteracion < iteraciones; ++iteracion) {
			Imagen siguiente = estado;

			for (std::size_t i = 0; i < dimension_; ++i) {
				double suma = 0.0;
				const std::size_t fila = i * dimension_;
				for (std::size_t j = 0; j < dimension_; ++j) {
					suma += pesos_[fila + j] * estado[j];
				}
				siguiente[i] = suma >= 0.0 ? 1.0 : -1.0;
			}

			if (siguiente == estado) {
				break;
			}

			estado.swap(siguiente);
		}

		return estado;
	}

private:
	std::size_t dimension_;
	std::vector<double> pesos_;
};

} // namespace bi_transformer

#endif