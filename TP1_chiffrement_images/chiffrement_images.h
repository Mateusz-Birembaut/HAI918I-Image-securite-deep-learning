#ifndef CHIFFREMENT_H
#define CHIFFREMENT_H





#include "../src/image_ppm.h"
#include "../src/random.h"
#include "../src/traitement.h"

#include <set>

typedef unsigned char OCTET;

void permutation(char cNomImgLue[256], char cNomImgSortie[256], char key) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	std::set<int> indexFilled{};

	std::mt19937 rng(static_cast<unsigned int>(key));
	std::uniform_int_distribution<int> dist(0, nTaille - 1);

	for (int i{0}; i < nTaille; i++) {
		int newIndex = dist(rng);
		while (indexFilled.find(newIndex) != indexFilled.end()) {
			++newIndex;
			if (newIndex == nTaille)
				newIndex = 0;
		}
		ImgOut[newIndex] = ImgIn[i];
		indexFilled.insert(newIndex);
	}

	ecrire_image_pgm(cNomImgSortie, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void permutation_inv(char cNomImgLue[256], char cNomImgSortie[256], char key) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	std::set<int> indexFilled{};
	std::vector<int> permutation(nTaille);

	std::mt19937 rng(static_cast<unsigned int>(key));
	std::uniform_int_distribution<int> dist(0, nTaille - 1);

	for (int i{0}; i < nTaille; i++) {
		int newIndex = dist(rng);
		while (indexFilled.find(newIndex) != indexFilled.end()) {
			++newIndex;
			if (newIndex == nTaille)
				newIndex = 0;
		}

		indexFilled.insert(newIndex);
		permutation[i] = newIndex;
	}

	for (int i{0}; i < nTaille; i++) {
		ImgOut[i] = ImgIn[permutation[i]];
	}

	ecrire_image_pgm(cNomImgSortie, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void substitution(char cNomImgLue[256], char cNomImgSortie[256], char key) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	std::vector<int> k(nTaille);

	std::mt19937 rng(static_cast<unsigned int>(key));
	std::uniform_int_distribution<int> dist(0, nTaille - 1);

	for (int i = 0; i < nTaille; i++) {
		k[i] = dist(rng);
	}

	for (int i{0}; i < nTaille; i++) {
		ImgOut[i] = (ImgIn[i] + k[i]) % 256;
	}

	ecrire_image_pgm(cNomImgSortie, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void substitution_inv(char cNomImgLue[256], char cNomImgSortie[256], char key) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	std::vector<int> k(nTaille);

	std::mt19937 rng(static_cast<unsigned int>(key));
	std::uniform_int_distribution<int> dist(0, nTaille - 1);

	for (int i = 0; i < nTaille; i++) {
		k[i] = dist(rng);
	}

	for (int i{0}; i < nTaille; i++) {
		ImgOut[i] = (ImgIn[i] - k[i]) % 256;
	}

	ecrire_image_pgm(cNomImgSortie, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

double entropie_img(char cNomImgLue[250]) {
	int nH, nW, nTaille;
	OCTET* ImgIn;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);

	int histo[256] = {0};
	for (int i = 0; i < nTaille; i++) {
		histo[ImgIn[i]]++;
	}

	double entropy = 0.0;
	for (int i = 0; i < 256; i++) {
		if (histo[i] > 0) {
			double p = (double)histo[i] / nTaille;
			entropy -= p * log2(p);
		}
	}

	free(ImgIn);
	return entropy;
}

void brut_force_substitution(char cNomImgLue[256], char cNomImgSortie[256]) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut, *ImgTemp;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);
	allocation_tableau(ImgTemp, OCTET, nTaille);

	char cNomTemp[]{"temp_brut_force"};

	double entropieMin{std::numeric_limits<double>::max()};
	int key{};

	double entropieTemp;

	for (int i = 0; i < 256; ++i) {
		substitution_inv(cNomImgLue, cNomTemp, i);
		entropieTemp = entropie_img(cNomTemp);
		if (entropieTemp < entropieMin) {
			entropieMin = entropieTemp;
			key = i;
		}
	}

	std::cout << "La clé est : " << key << '\n';
	std::cout << "L'entropie est de : " << entropieMin << '\n';

    free(ImgOut);
    free(ImgIn);
}

#endif // CHIFFREMENT_H