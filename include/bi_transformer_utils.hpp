#ifndef BI_TRANSFORMER_UTILS_HPP
#define BI_TRANSFORMER_UTILS_HPP

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace bi_transformer {

using Imagen = std::vector<double>;
using ConjuntoImagenes = std::vector<Imagen>;
using Etiquetas = std::vector<std::uint8_t>;

inline double limitar(double valor, double minimo, double maximo) {
	return std::max(minimo, std::min(maximo, valor));
}

inline double distancia_cuadrada(const Imagen& a, const Imagen& b) {
	double suma = 0.0;
	for (std::size_t i = 0; i < a.size(); ++i) {
		const double diferencia = a[i] - b[i];
		suma += diferencia * diferencia;
	}
	return suma;
}

inline void normalizar_imagenes(ConjuntoImagenes& imagenes) {
	for (auto& imagen : imagenes) {
		for (auto& pixel : imagen) {
			pixel = limitar(pixel / 255.0, 0.0, 1.0);
		}
	}
}

inline Imagen convertir_a_bipolar(const Imagen& valores, double umbral = 0.25) {
	Imagen resultado;
	resultado.reserve(valores.size());

	for (double valor : valores) {
		resultado.push_back(valor >= umbral ? 1.0 : -1.0);
	}

	return resultado;
}

inline std::size_t parsear_tamano(const char* texto, std::size_t valor_por_defecto) {
	if (!texto || !*texto) {
		return valor_por_defecto;
	}

	try {
		return static_cast<std::size_t>(std::stoull(texto));
	} catch (...) {
		return valor_por_defecto;
	}
}

} // namespace bi_transformer

#endif