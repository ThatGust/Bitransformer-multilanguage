#ifndef MAPA_KOHONEN_HPP
#define MAPA_KOHONEN_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "bi_transformer_utils.hpp"

namespace bi_transformer {

class MapaKohonen {
public:
	MapaKohonen(std::size_t filas, std::size_t columnas, std::size_t dimension)
		: filas_(filas), columnas_(columnas), dimension_(dimension), pesos_(filas * columnas, Imagen(dimension, 0.0)), generador_(12345u) {
		std::uniform_real_distribution<double> distribucion(0.0, 1.0);
		for (auto& peso : pesos_) {
			for (auto& valor : peso) {
				valor = distribucion(generador_);
			}
		}
	}

	void entrenar(const ConjuntoImagenes& muestras, std::size_t epocas) {
		if (muestras.empty() || epocas == 0) {
			return;
		}

		const std::size_t maximo_muestras = std::min<std::size_t>(6000, muestras.size());
		std::vector<std::size_t> indices(muestras.size());
		std::iota(indices.begin(), indices.end(), 0);

		const double tasa_inicial = 0.5;
		const double tasa_final = 0.05;
		const double sigma_inicial = std::max(filas_, columnas_) / 2.0;
		const double sigma_final = 1.0;

		for (std::size_t epoca = 0; epoca < epocas; ++epoca) {
			std::cout << "\rEntrenando BiTransformer - epoca " << (epoca + 1) << "/" << epocas << std::flush;
			std::shuffle(indices.begin(), indices.end(), generador_);

			const double progreso = epocas == 1 ? 1.0 : static_cast<double>(epoca) / static_cast<double>(epocas - 1);
			const double tasa = tasa_inicial * std::pow(tasa_final / tasa_inicial, progreso);
			const double sigma = sigma_inicial * std::pow(sigma_final / sigma_inicial, progreso);

			for (std::size_t pos = 0; pos < maximo_muestras; ++pos) {
				const Imagen& muestra = muestras[indices[pos]];
				const std::size_t ganador = buscar_ganador(muestra);
				actualizar(muestra, ganador, tasa, sigma);
			}
		}

		std::cout << "\rEntrenando BiTransformer - epoca " << epocas << "/" << epocas << std::endl;
	}

	Imagen transformar(const Imagen& entrada) const {
		const std::size_t ganador = buscar_ganador(entrada);
		Imagen activaciones(pesos_.size(), 0.0);

		const int fila_ganadora = static_cast<int>(ganador / columnas_);
		const int columna_ganadora = static_cast<int>(ganador % columnas_);
		const double sigma = 1.0;
		const double divisor = 2.0 * sigma * sigma;

		for (std::size_t unidad = 0; unidad < pesos_.size(); ++unidad) {
			const int fila = static_cast<int>(unidad / columnas_);
			const int columna = static_cast<int>(unidad % columnas_);
			const double df = static_cast<double>(fila - fila_ganadora);
			const double dc = static_cast<double>(columna - columna_ganadora);
			activaciones[unidad] = std::exp(-(df * df + dc * dc) / divisor);
		}

		return activaciones;
	}

	std::size_t dimension_salida() const {
		return pesos_.size();
	}

private:
	std::size_t filas_;
	std::size_t columnas_;
	std::size_t dimension_;
	ConjuntoImagenes pesos_;
	std::mt19937 generador_;

	std::size_t buscar_ganador(const Imagen& entrada) const {
		std::size_t mejor_indice = 0;
		double mejor_distancia = std::numeric_limits<double>::max();

		for (std::size_t unidad = 0; unidad < pesos_.size(); ++unidad) {
			const double distancia = distancia_cuadrada(entrada, pesos_[unidad]);
			if (distancia < mejor_distancia) {
				mejor_distancia = distancia;
				mejor_indice = unidad;
			}
		}

		return mejor_indice;
	}

	void actualizar(const Imagen& entrada, std::size_t ganador, double tasa, double sigma) {
		const int fila_ganadora = static_cast<int>(ganador / columnas_);
		const int columna_ganadora = static_cast<int>(ganador % columnas_);
		const double divisor = 2.0 * sigma * sigma;

		for (std::size_t unidad = 0; unidad < pesos_.size(); ++unidad) {
			const int fila = static_cast<int>(unidad / columnas_);
			const int columna = static_cast<int>(unidad % columnas_);
			const double df = static_cast<double>(fila - fila_ganadora);
			const double dc = static_cast<double>(columna - columna_ganadora);
			const double influencia = std::exp(-(df * df + dc * dc) / divisor);

			if (influencia < 1e-6) {
				continue;
			}

			Imagen& peso = pesos_[unidad];
			for (std::size_t i = 0; i < dimension_; ++i) {
				peso[i] += tasa * influencia * (entrada[i] - peso[i]);
			}
		}
	}
};

} // namespace bi_transformer

#endif