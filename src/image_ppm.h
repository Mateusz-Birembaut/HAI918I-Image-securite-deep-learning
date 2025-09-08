// ----------------------------------------------------------------------------
// Filename        : image_ppm.c
// Description     :
// Created On      : Tue Mar 31 13:26:36 2005
// ----------------------------------------------------------------------------

#ifndef IMAGE_H
#define IMAGE_H





#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define allocation_tableau(nom, type, nombre)                                             \
	if ((nom = (type*)calloc(nombre, sizeof(type))) == NULL) {                        \
		printf("\n Allocation dynamique impossible pour un pointeur-tableau \n"); \
		exit(1);                                                                  \
	}

typedef unsigned char OCTET;

/*===========================================================================*/
void ignorer_commentaires(FILE* f) {
	unsigned char c;
	while ((c = fgetc(f)) == '#')
		while ((c = fgetc(f)) != '\n')
			;
	fseek(f, -sizeof(unsigned char), SEEK_CUR);
}
/*===========================================================================*/

/*===========================================================================*/
void ecrire_image_ppm(char nom_image[], OCTET* pt_image, int nb_lignes, int nb_colonnes) {
	FILE* f_image;
	int taille_image = 3 * nb_colonnes * nb_lignes;

	if ((f_image = fopen(nom_image, "wb")) == NULL) {
		printf("\nPas d'acces en ecriture sur l'image %s \n", nom_image);
		exit(EXIT_FAILURE);
	} else {
		fprintf(f_image, "P6\r"); /*ecriture entete*/
		fprintf(f_image, "%d %d\r255\r", nb_colonnes, nb_lignes);

		if ((fwrite((OCTET*)pt_image, sizeof(OCTET), taille_image, f_image)) != (size_t)(taille_image)) {
			printf("\nErreur d'ecriture de l'image %s \n", nom_image);
			exit(EXIT_FAILURE);
		}
		fclose(f_image);
	}
}
/*===========================================================================*/

/*===========================================================================*/
void lire_nb_lignes_colonnes_image_ppm(char nom_image[], int* nb_lignes, int* nb_colonnes) {
	FILE* f_image;
	int max_grey_val;

	/* cf : l'entete d'une image .ppm : P6                   */
	/*				       nb_colonnes nb_lignes */
	/*    			       max_grey_val          */

	if ((f_image = fopen(nom_image, "rb")) == NULL) {
		printf("\nPas d'acces en lecture sur l'image %s \n", nom_image);
		exit(EXIT_FAILURE);
	} else {
		fscanf(f_image, "P6 ");
		ignorer_commentaires(f_image);
		fscanf(f_image, "%d %d %d%*c", nb_colonnes, nb_lignes, &max_grey_val);
		fclose(f_image);
	}
}
/*===========================================================================*/
/*===========================================================================*/
void lire_image_ppm(char nom_image[], OCTET* pt_image, int taille_image) {
	FILE* f_image;
	int nb_colonnes, nb_lignes, max_grey_val;
	taille_image = 3 * taille_image;

	if ((f_image = fopen(nom_image, "rb")) == NULL) {
		printf("\nPas d'acces en lecture sur l'image %s \n", nom_image);
		exit(EXIT_FAILURE);
	} else {
		fscanf(f_image, "P6 ");
		ignorer_commentaires(f_image);
		fscanf(f_image, "%d %d %d%*c",
		       &nb_colonnes, &nb_lignes, &max_grey_val); /*lecture entete*/

		if ((fread((OCTET*)pt_image, sizeof(OCTET), taille_image, f_image)) != (size_t)(taille_image)) {
			printf("\nErreur de lecture de l'image %s \n", nom_image);
			exit(EXIT_FAILURE);
		}
		fclose(f_image);
	}
}

/*===========================================================================*/
/*===========================================================================*/

void planR(OCTET* pt_image, OCTET* src, int taille_image) {
	int i;
	for (i = 0; i < taille_image; i++) {
		pt_image[i] = src[3 * i];
	}
}

/*===========================================================================*/
/*===========================================================================*/

void planV(OCTET* pt_image, OCTET* src, int taille_image) {
	int i;
	for (i = 0; i < taille_image; i++) {
		pt_image[i] = src[3 * i + 1];
	}
}

/*===========================================================================*/
/*===========================================================================*/

void planB(OCTET* pt_image, OCTET* src, int taille_image) {
	int i;
	for (i = 0; i < taille_image; i++) {
		pt_image[i] = src[3 * i + 2];
	}
}

/*===========================================================================*/
/*===========================================================================*/

void ecrire_image_pgm(char nom_image[], OCTET* pt_image, int nb_lignes, int nb_colonnes) {
	FILE* f_image;
	int taille_image = nb_colonnes * nb_lignes;

	if ((f_image = fopen(nom_image, "wb")) == NULL) {
		printf("\nPas d'acces en ecriture sur l'image %s \n", nom_image);
		exit(EXIT_FAILURE);
	} else {
		fprintf(f_image, "P5\r"); /*ecriture entete*/
		fprintf(f_image, "%d %d\r255\r", nb_colonnes, nb_lignes);

		if ((fwrite((OCTET*)pt_image, sizeof(OCTET), taille_image, f_image)) != (size_t)taille_image) {
			printf("\nErreur de lecture de l'image %s \n", nom_image);
			exit(EXIT_FAILURE);
		}
		fclose(f_image);
	}
}
/*===========================================================================*/

void lire_nb_lignes_colonnes_image_pgm(char nom_image[], int* nb_lignes, int* nb_colonnes) {
	FILE* f_image;
	int max_grey_val;

	/* cf : l'entete d'une image .pgm : P5                    */
	/*				       nb_colonnes nb_lignes */
	/*    			       max_grey_val          */

	if ((f_image = fopen(nom_image, "rb")) == NULL) {
		printf("\nPas d'acces en lecture sur l'image %s \n", nom_image);
		exit(EXIT_FAILURE);
	} else {
		fscanf(f_image, "P5 ");
		ignorer_commentaires(f_image);
		fscanf(f_image, "%d %d %d%*c", nb_colonnes, nb_lignes, &max_grey_val);
		fclose(f_image);
	}
}
/*===========================================================================*/
/*===========================================================================*/
void lire_image_pgm(char nom_image[], OCTET* pt_image, int taille_image) {
	FILE* f_image;
	int nb_colonnes, nb_lignes, max_grey_val;

	if ((f_image = fopen(nom_image, "rb")) == NULL) {
		printf("\nPas d'acces en lecture sur l'image %s \n", nom_image);
		exit(EXIT_FAILURE);
	} else {
		fscanf(f_image, "P5 ");
		ignorer_commentaires(f_image);
		fscanf(f_image, "%d %d %d%*c",
		       &nb_colonnes, &nb_lignes, &max_grey_val); /*lecture entete*/

		if ((fread((OCTET*)pt_image, sizeof(OCTET), taille_image, f_image)) != (size_t)taille_image) {
			printf("\nErreur de lecture de l'image %s \n", nom_image);
			exit(EXIT_FAILURE);
		}
		fclose(f_image);
	}
}

/*============================== utils =============================================*/

int getValeurMoyenne(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon) {
	int moyenne = 0;
	int nb_voisins = 0;

	for (int i = -rayon; i <= rayon; i++) {
		for (int j = -rayon; j <= rayon; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;

			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
				moyenne = moyenne + image[voisinLigne * nW + voisinColonne];
				nb_voisins++;
			}
		}
	}
	return moyenne / nb_voisins;
}

int getCouleurMinimal(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon) {
	int minimum = image[ligne * nW + colonne];
	for (int i = -rayon; i <= rayon; i++) {
		for (int j = -rayon; j <= rayon; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;
			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
				int valeurVoisin = image[voisinLigne * nW + voisinColonne];
				if (valeurVoisin < minimum) {
					minimum = valeurVoisin;
				}
			}
		}
	}
	return minimum;
}

int getCouleurMaximal(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon) {
	int maximum = image[ligne * nW + colonne];
	for (int i = -rayon; i <= rayon; i++) {
		for (int j = -rayon; j <= rayon; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;
			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
				int valeurVoisin = image[voisinLigne * nW + voisinColonne];
				if (valeurVoisin > maximum) {
					maximum = valeurVoisin;
				}
			}
		}
	}
	return maximum;
}

int getPixelValueWithMirrorPadding(OCTET* image, int nW, int nH, int ligne, int colonne) { // si on veut utiliser mirror padding, remplacer valeurVoisin = getPixelValueWithMirrorPadding et enlever if
	if (ligne < 0)
		ligne = -ligne;
	else if (ligne >= nH)
		ligne = 2 * nH - ligne - 1;
	if (colonne < 0)
		colonne = -colonne;
	else if (colonne >= nW)
		colonne = 2 * nW - colonne - 1;

	return image[ligne * nW + colonne];
}

int getNormeGradient(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon, int masqueHorizontal[], int masqueVertical[]) {
	int gradientHorizontal = 0;
	int gradientVertical = 0;
	for (int i = -rayon; i <= rayon; i++) {
		for (int j = -rayon; j <= rayon; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;
			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
				int valeurVoisin = image[voisinLigne * nW + voisinColonne];
				gradientHorizontal += valeurVoisin * masqueHorizontal[(i + rayon) * 3 + (j + rayon)];
				gradientVertical += valeurVoisin * masqueVertical[(i + rayon) * 3 + (j + rayon)];
			}
		}
	}
	return sqrt(pow(gradientHorizontal, 2) + pow(gradientVertical, 2));
	// return abs(gradientHorizontal) + abs(gradientVertical);
}

int checkVoisinsHysteresis(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon, int seuilBas, int seuilHaut) {
	int valeur = image[ligne * nW + colonne];
	if (valeur > seuilBas && valeur < seuilHaut) {
		for (int i = -rayon; i <= rayon; i++) {
			for (int j = -rayon; j <= rayon; j++) {
				int voisinLigne = ligne + i;
				int voisinColonne = colonne + j;
				if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
					int valeurVoisin = image[voisinLigne * nW + voisinColonne];
					if (valeurVoisin == 255) {
						return 255;
					}
				}
			}
		}
		return 0;
	}
	return valeur;
}

int checkVoisinsHysteresisV2(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon, int seuilBas, int seuilHaut) {
	int valeur = image[ligne * nW + colonne];
	if (valeur > seuilBas && valeur < seuilHaut) {
		for (int i = -rayon; i <= rayon; i++) {
			for (int j = -rayon; j <= rayon; j++) {
				int voisinLigne = ligne + i;
				int voisinColonne = colonne + j;
				if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
					int valeurVoisin = image[voisinLigne * nW + voisinColonne];
					if (valeurVoisin == 255) {
						return 255;
					}
				}
			}
		}
		return 0;
	}
	return valeur;
}

int compare(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}

int getValeurMediane(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon) {
	int voisins[(2 * rayon + 1) * (2 * rayon + 1)];
	int compteur = 0;
	for (int i = -rayon; i <= rayon; i++) {
		for (int j = -rayon; j <= rayon; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;
			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
				voisins[compteur] = image[voisinLigne * nW + voisinColonne];
				compteur++;
			}
		}
	}
	// sort array
	qsort(voisins, compteur, sizeof(int), compare);

	if (compteur % 2 == 0) {
		return (voisins[compteur / 2 - 1] + voisins[compteur / 2]) / 2;
	} else { // Si le nombre de voisins est impair, prendre simplement la valeur du milieu
		return voisins[compteur / 2];
	}
}

void filtreMoyenneurPGM(OCTET* image, int nW, int nH, int rayon) {
	int nTaille;

	OCTET* ImgOut;

	nTaille = nH * nW;
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = getValeurMediane(image, nW, nH, i, j, rayon);
		}
	}

	for (int i = 0; i < nTaille; i++) {
		image[i] = ImgOut[i];
	}
}

int getValeurGauss(OCTET* image, int nW, int nH, int ligne, int colonne, int rayon) {
	int masqueGauss[] = {1, 2, 1,
			     2, 4, 2,
			     1, 2, 1};
	int value = 0;
	int sommeMasque = 0;

	for (int i = -rayon; i <= rayon; i++) {
		for (int j = -rayon; j <= rayon; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;
			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
				value += (image[voisinLigne * nW + voisinColonne] * masqueGauss[(i + rayon) * 3 + (j + rayon)]);
				sommeMasque += masqueGauss[(i + rayon) * (2 * rayon + 1) + (j + rayon)];
			}
		}
	}

	if (sommeMasque != 0) {
		value /= sommeMasque;
	}

	return value;
}

void filtreGaussienPGM(OCTET* image, int nW, int nH, int rayon) {
	int nTaille;

	OCTET* ImgOut;

	nTaille = nH * nW;
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = getValeurGauss(image, nW, nH, i, j, rayon);
		}
	}

	for (int i = 0; i < nTaille; i++) {
		image[i] = ImgOut[i];
	}
}

OCTET clamp(double valeur) {
	if (valeur < 0)
		return 0;
	if (valeur > 255)
		return 255;
	return (OCTET)valeur;
}

/*============================== TP 1 =============================================*/

void seuil_image_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int seuils[], int nb_Seuils) {
	int nH, nW, nTaille;

	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			int lvl = 0;
			for (int k = 0; k < nb_Seuils; k++) {
				if (ImgIn[i * nW + j] >= seuils[k]) {
					lvl++;
				}
			}
			ImgOut[i * nW + j] = lvl * (255 / nb_Seuils);
		}
	}

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void seuil_image_pgm_spe(char cNomImgLue[250], char cNomImgEcrite[250], int seuil) {
	int nH, nW, nTaille;

	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = (ImgIn[i * nW + j] >= seuil) ? 255 : 0;
		}
	}

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void seuil_image_ppm(char cNomImgLue[250], char cNomImgEcrite[250]) {
	int nH, nW, nTaille;
	int nTaille3;

	OCTET *ImgIn, *ImgOut, *PlanR, *PlanV, *PlanB;

	// Lire la taille de l'image
	lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;
	nTaille3 = nTaille * 3;

	// Allocation mémoire pour l'image d'entrée et les plans R, V, B
	allocation_tableau(ImgIn, OCTET, nTaille3);
	lire_image_ppm(cNomImgLue, ImgIn, nTaille);

	allocation_tableau(PlanR, OCTET, nTaille);
	planR(PlanR, ImgIn, nTaille);

	allocation_tableau(PlanV, OCTET, nTaille);
	planV(PlanV, ImgIn, nTaille);

	allocation_tableau(PlanB, OCTET, nTaille);
	planB(PlanB, ImgIn, nTaille);

	// Demander à l'utilisateur le nombre de seuils pour chaque composante
	int nb_SeuilsR, nb_SeuilsV, nb_SeuilsB;
	printf("Entrez le nombre de seuils pour la composante Rouge (0 pour pas de seuil) : ");
	scanf("%d", &nb_SeuilsR);

	printf("Entrez le nombre de seuils pour la composante Verte (0 pour pas de seuil) : ");
	scanf("%d", &nb_SeuilsV);

	printf("Entrez le nombre de seuils pour la composante Bleue (0 pour pas de seuil) : ");
	scanf("%d", &nb_SeuilsB);

	// Allocation des tableaux de seuils pour chaque composante
	int *seuilsR = NULL, *seuilsV = NULL, *seuilsB = NULL;

	// Si le nombre de seuils est supérieur à 0, allouer et demander les seuils
	if (nb_SeuilsR > 0) {
		seuilsR = (int*)malloc(nb_SeuilsR * sizeof(int));
		printf("Entrez les seuils pour la composante Rouge :\n");
		for (int i = 0; i < nb_SeuilsR; i++) {
			printf("Seuil %d : ", i + 1);
			scanf("%d", &seuilsR[i]);
		}
	}

	if (nb_SeuilsV > 0) {
		seuilsV = (int*)malloc(nb_SeuilsV * sizeof(int));
		printf("Entrez les seuils pour la composante Verte :\n");
		for (int i = 0; i < nb_SeuilsV; i++) {
			printf("Seuil %d : ", i + 1);
			scanf("%d", &seuilsV[i]);
		}
	}

	if (nb_SeuilsB > 0) {
		seuilsB = (int*)malloc(nb_SeuilsB * sizeof(int));
		printf("Entrez les seuils pour la composante Bleue :\n");
		for (int i = 0; i < nb_SeuilsB; i++) {
			printf("Seuil %d : ", i + 1);
			scanf("%d", &seuilsB[i]);
		}
	}

	// Traitement des seuils pour chaque composante (R, V, B)
	for (int i = 0; i < nTaille; i++) {
		// Traitement pour la composante Rouge (R)
		if (nb_SeuilsR > 0) {
			int lvlR = 0;
			for (int k = 0; k < nb_SeuilsR; k++) {
				if (PlanR[i] >= seuilsR[k]) {
					lvlR++;
				}
			}
			PlanR[i] = lvlR * (255 / nb_SeuilsR);
		}

		// Traitement pour la composante Verte (V)
		if (nb_SeuilsV > 0) {
			int lvlV = 0;
			for (int k = 0; k < nb_SeuilsV; k++) {
				if (PlanV[i] >= seuilsV[k]) {
					lvlV++;
				}
			}
			PlanV[i] = lvlV * (255 / nb_SeuilsV);
		}

		// Traitement pour la composante Bleue (B)
		if (nb_SeuilsB > 0) {
			int lvlB = 0;
			for (int k = 0; k < nb_SeuilsB; k++) {
				if (PlanB[i] >= seuilsB[k]) {
					lvlB++;
				}
			}
			PlanB[i] = lvlB * (255 / nb_SeuilsB);
		}
	}

	// Allocation de l'image de sortie
	allocation_tableau(ImgOut, OCTET, nTaille3);
	for (int i = 0; i < nTaille; i++) {
		ImgOut[i * 3] = PlanR[i];
		ImgOut[i * 3 + 1] = PlanV[i];
		ImgOut[i * 3 + 2] = PlanB[i];
	}

	// Écriture de l'image de sortie
	ecrire_image_ppm(cNomImgEcrite, ImgOut, nH, nW);

	// Libération de la mémoire
	free(ImgIn);
	free(ImgOut);
	free(PlanR);
	free(PlanV);
	free(PlanB);
	if (seuilsR)
		free(seuilsR);
	if (seuilsV)
		free(seuilsV);
	if (seuilsB)
		free(seuilsB);
}

void ecrire_histogramme_pgm(char nom_fichier[250], int histogramme[], int taille_histogramme) {
	FILE* fichier;

	if ((fichier = fopen(nom_fichier, "w")) == NULL) {
		printf("\nPas d'acces en ecriture sur le fichier %s \n", nom_fichier);
		exit(EXIT_FAILURE);
	} else {
		fprintf(fichier, "%s %s\n", "Indice", "Niveau Gris");
		for (int i = 0; i < taille_histogramme; i++) {
			fprintf(fichier, "%d %d\n", i, histogramme[i]);
		}

		fclose(fichier);
	}
}

void histogramme_pgm(char cNomImgLue[250], char cNomFichierEcrit[250]) {
	int nH, nW, nTaille;
	OCTET* ImgIn;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);

	int histogramme[256] = {0};
	for (int i = 0; i < nTaille; i++) {
		histogramme[ImgIn[i]]++;
	}

	ecrire_histogramme_pgm(cNomFichierEcrit, histogramme, 256);
	free(ImgIn);
}

void ecrire_profil(const char* nom_fichier, int profil[], int tailleProfil) {
	FILE* fichier;

	if ((fichier = fopen(nom_fichier, "w")) == NULL) {
		printf("\nPas d'accès en écriture sur le fichier %s \n", nom_fichier);
		exit(EXIT_FAILURE);
	} else {
		for (int i = 0; i < tailleProfil; i++) {
			fprintf(fichier, "%d %d\n", i, profil[i]);
		}
		fclose(fichier);
	}
}

void profil_ligne_pgm(char cNomImgLue[250], int indice, char cNomFichierEcrit[250]) {
	int nH, nW;
	OCTET* ImgIn;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	allocation_tableau(ImgIn, OCTET, nH * nW);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);

	int profil[nW];
	for (int i = 0; i < nW; i++) {
		profil[i] = ImgIn[indice * nW + i];
	}
	ecrire_profil(cNomFichierEcrit, profil, nW);

	free(ImgIn);
}

void profil_colonne_pgm(char cNomImgLue[250], int indice, char cNomFichierEcrit[250]) {
	int nH, nW;
	OCTET* ImgIn;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	allocation_tableau(ImgIn, OCTET, nH * nW);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);

	int profil[nH];
	for (int i = 0; i < nH; i++) {
		profil[i] = ImgIn[i * nW + indice];
	}
	ecrire_profil(cNomFichierEcrit, profil, nH);

	free(ImgIn);
}

void ecrire_histogramme_ppm(const char* nom_fichier, int histogramme[][256], int taille_histogramme) {
	FILE* fichier;

	if ((fichier = fopen(nom_fichier, "w")) == NULL) {
		printf("\nPas d'accès en écriture sur le fichier %s \n", nom_fichier);
		exit(EXIT_FAILURE);
	} else {
		fprintf(fichier, "Indice R G B\n");
		for (int i = 0; i < taille_histogramme; i++) {
			fprintf(fichier, "%d %d %d %d\n", i, histogramme[0][i], histogramme[1][i], histogramme[2][i]);
		}
		fclose(fichier);
	}
}

void histogramme_ppm(char cNomImgLue[250], char cNomFichierEcrit[250]) {
	int nH, nW, nTaille;
	OCTET* ImgIn;

	lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW * 3;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_ppm(cNomImgLue, ImgIn, nTaille / 3);
	int histogramme[3][256] = {{0}};

	for (int i = 0; i < nTaille; i += 3) {
		histogramme[0][ImgIn[i]]++;
		histogramme[1][ImgIn[i + 1]]++;
		histogramme[2][ImgIn[i + 2]]++;
	}

	ecrire_histogramme_ppm(cNomFichierEcrit, histogramme, 256);
	free(ImgIn);
}

/*============================== TP 2 =============================================*/

void erosion(OCTET* in, OCTET* out, int nH, int nW, int rayon) {

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			out[i * nW + j] = getCouleurMinimal(in, nW, nH, i, j, rayon);
		}
	}
}

void erosion_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	erosion(ImgIn, ImgOut, nH, nW, rayon);

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void erosion_ppm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	int nH, nW, nTaille;

	lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;
	int nTaille3 = nTaille * 3;

	OCTET *ImgIn, *ImgOut, *PlanR, *PlanV, *PlanB;

	allocation_tableau(ImgIn, OCTET, nTaille3);
	lire_image_ppm(cNomImgLue, ImgIn, nH * nW);

	allocation_tableau(PlanR, OCTET, nTaille);
	planR(PlanR, ImgIn, nTaille);

	allocation_tableau(PlanV, OCTET, nTaille);
	planV(PlanV, ImgIn, nTaille);

	allocation_tableau(PlanB, OCTET, nTaille);
	planB(PlanB, ImgIn, nTaille);

	OCTET *PlanRTemp, *PlanVTemp, *PlanBTemp;

	allocation_tableau(PlanRTemp, OCTET, nTaille);

	allocation_tableau(PlanVTemp, OCTET, nTaille);

	allocation_tableau(PlanBTemp, OCTET, nTaille);

	allocation_tableau(ImgOut, OCTET, nTaille3);

	erosion(PlanR, PlanRTemp, nH, nW, rayon);
	erosion(PlanV, PlanVTemp, nH, nW, rayon);
	erosion(PlanB, PlanBTemp, nH, nW, rayon);

	for (int i = 0; i < nTaille; i++) {
		ImgOut[i * 3] = PlanRTemp[i];
		ImgOut[i * 3 + 1] = PlanVTemp[i];
		ImgOut[i * 3 + 2] = PlanBTemp[i];
	}

	ecrire_image_ppm(cNomImgEcrite, ImgOut, nH, nW);

	free(PlanR);
	free(PlanV);
	free(PlanB);
	free(ImgIn);
	free(PlanRTemp);
	free(PlanVTemp);
	free(PlanBTemp);
}

void dilatation(OCTET* in, OCTET* out, int nH, int nW, int rayon) {
	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			out[i * nW + j] = getCouleurMaximal(in, nW, nH, i, j, rayon);
		}
	}
}

void dilatation_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	dilatation(ImgIn, ImgOut, nH, nW, rayon);

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void dilatation_ppm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	int nH, nW, nTaille;

	lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;
	int nTaille3 = nTaille * 3;

	OCTET *ImgIn, *ImgOut, *PlanR, *PlanV, *PlanB;

	allocation_tableau(ImgIn, OCTET, nTaille3);
	lire_image_ppm(cNomImgLue, ImgIn, nH * nW);

	allocation_tableau(PlanR, OCTET, nTaille);
	planR(PlanR, ImgIn, nTaille);

	allocation_tableau(PlanV, OCTET, nTaille);
	planV(PlanV, ImgIn, nTaille);

	allocation_tableau(PlanB, OCTET, nTaille);
	planB(PlanB, ImgIn, nTaille);

	OCTET *PlanRTemp, *PlanVTemp, *PlanBTemp;

	allocation_tableau(PlanRTemp, OCTET, nTaille);

	allocation_tableau(PlanVTemp, OCTET, nTaille);

	allocation_tableau(PlanBTemp, OCTET, nTaille);

	allocation_tableau(ImgOut, OCTET, nTaille3);

	dilatation(PlanR, PlanRTemp, nH, nW, rayon);
	dilatation(PlanV, PlanVTemp, nH, nW, rayon);
	dilatation(PlanB, PlanBTemp, nH, nW, rayon);

	for (int i = 0; i < nTaille; i++) {
		ImgOut[i * 3] = PlanRTemp[i];
		ImgOut[i * 3 + 1] = PlanVTemp[i];
		ImgOut[i * 3 + 2] = PlanBTemp[i];
	}

	ecrire_image_ppm(cNomImgEcrite, ImgOut, nH, nW);

	free(PlanR);
	free(PlanV);
	free(PlanB);
	free(ImgIn);
	free(PlanRTemp);
	free(PlanVTemp);
	free(PlanBTemp);
}

void fermeture_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	char cNomImgTemp[250] = "temp.pgm"; // Nom temporaire pour le fichier intermédiaire

	// Appliquer l'érosion, puis la dilatation sur l'image
	erosion_pgm(cNomImgLue, cNomImgTemp, rayon);	   // Érosion et sauvegarde dans un fichier temporaire
	dilatation_pgm(cNomImgTemp, cNomImgEcrite, rayon); // Dilatation du résultat de l'érosion et sauvegarde dans le fichier final

	remove(cNomImgTemp);
}

void fermeture_ppm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	char cNomImgTemp[250] = "temp.pgm";

	erosion_ppm(cNomImgLue, cNomImgTemp, rayon);
	dilatation_ppm(cNomImgTemp, cNomImgEcrite, rayon);

	remove(cNomImgTemp);
}

void ouverture_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	char cNomImgTemp[250] = "temp.pgm"; // Nom temporaire pour le fichier intermédiaire

	// Appliquer la dilatation sur l'image, puis l'érosion,
	dilatation_pgm(cNomImgLue, cNomImgTemp, rayon); // Dilatation du résultat de l'érosion et sauvegarde dans le fichier final
	erosion_pgm(cNomImgTemp, cNomImgEcrite, rayon); // Érosion et sauvegarde dans un fichier temporaire

	remove(cNomImgTemp);
}

void ouverture_ppm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	char cNomImgTemp[250] = "temp.pgm";

	dilatation_ppm(cNomImgLue, cNomImgTemp, rayon);
	erosion_ppm(cNomImgTemp, cNomImgEcrite, rayon);

	remove(cNomImgTemp);
}

void difference(OCTET* in1, OCTET* in2, OCTET* out, int nH, int nW) {
	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			out[i * nW + j] = abs(in2[i * nW + j] - in1[i * nW + j]);
		}
	}
}

void difference_pgm(char cNomImgLue1[250], char cNomImgLue2[250], char cNomImgEcrite[250]) {
	int nH, nW, nTaille;
	OCTET *ImgIn1, *ImgIn2, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue1, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn1, OCTET, nTaille);
	lire_image_pgm(cNomImgLue1, ImgIn1, nH * nW);
	allocation_tableau(ImgIn2, OCTET, nTaille);
	lire_image_pgm(cNomImgLue2, ImgIn2, nH * nW);
	allocation_tableau(ImgOut, OCTET, nTaille);

	difference(ImgIn1, ImgIn2, ImgOut, nH, nW);

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn1);
	free(ImgIn2);
	free(ImgOut);
}

void difference_ppm(char cNomImgLue1[250], char cNomImgLue2[250], char cNomImgEcrite[250]) {
	int nH, nW, nTaille;

	lire_nb_lignes_colonnes_image_ppm(cNomImgLue1, &nH, &nW);
	nTaille = nH * nW;
	int nTaille3 = nTaille * 3;

	OCTET *ImgIn1, *ImgIn2, *ImgOut, *PlanR, *PlanV, *PlanB, *PlanR2, *PlanV2, *PlanB2;

	//--------------- IMAGE IN 1 -------------
	allocation_tableau(ImgIn1, OCTET, nTaille3);
	lire_image_ppm(cNomImgLue1, ImgIn1, nH * nW);

	//--------------- IMAGE IN 2 -------------
	allocation_tableau(ImgIn2, OCTET, nTaille3);
	lire_image_ppm(cNomImgLue2, ImgIn2, nH * nW);

	// cannaux

	allocation_tableau(PlanR, OCTET, nTaille);
	allocation_tableau(PlanV, OCTET, nTaille);
	allocation_tableau(PlanB, OCTET, nTaille);

	allocation_tableau(PlanR2, OCTET, nTaille);
	allocation_tableau(PlanV2, OCTET, nTaille);
	allocation_tableau(PlanB2, OCTET, nTaille);

	planR(PlanR, ImgIn1, nTaille);
	planV(PlanV, ImgIn1, nTaille);
	planB(PlanB, ImgIn1, nTaille);

	planR(PlanR2, ImgIn2, nTaille);
	planV(PlanV2, ImgIn2, nTaille);
	planB(PlanB2, ImgIn2, nTaille);

	// stockage temp dans des plans

	OCTET *PlanRTemp, *PlanVTemp, *PlanBTemp;

	allocation_tableau(PlanRTemp, OCTET, nTaille);

	allocation_tableau(PlanVTemp, OCTET, nTaille);

	allocation_tableau(PlanBTemp, OCTET, nTaille);

	allocation_tableau(ImgOut, OCTET, nTaille3);

	difference(PlanR, PlanR2, PlanRTemp, nH, nW);
	difference(PlanV, PlanV2, PlanVTemp, nH, nW);
	difference(PlanB, PlanB2, PlanBTemp, nH, nW);

	for (int i = 0; i < nTaille; i++) {
		ImgOut[i * 3] = PlanRTemp[i];
		ImgOut[i * 3 + 1] = PlanVTemp[i];
		ImgOut[i * 3 + 2] = PlanBTemp[i];
	}

	ecrire_image_ppm(cNomImgEcrite, ImgOut, nH, nW);

	free(PlanR);
	free(PlanV);
	free(PlanB);
	free(ImgIn1);
	free(ImgIn2);
	free(PlanRTemp);
	free(PlanVTemp);
	free(PlanBTemp);
	free(PlanR2);
	free(PlanV2);
	free(PlanB2);
}

/*============================== TP 3 =============================================*/

void inverse_pgm(char cNomImgLue[250], char cNomImgEcrite[250]) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = 255 - ImgIn[i * nW + j];
		}
	}

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void flouter_image(OCTET* pt_image, int nH, int nW, int rayon) {
	int nTaille;

	OCTET* ImgOut;

	nTaille = nH * nW;
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = getValeurMoyenne(pt_image, nW, nH, i, j, rayon);
		}
	}
	for (int i = 0; i < nTaille; i++) {
		pt_image[i] = ImgOut[i];
	}

	free(ImgOut);
}

void flouter_image_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = getValeurMoyenne(ImgIn, nW, nH, i, j, rayon);
		}
	}

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void flouter_image_ppm(char cNomImgLue[250], char cNomImgEcrite[250], int rayon) {
	int nH, nW, nTaille;

	lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;
	int nTaille3 = nTaille * 3;

	OCTET *ImgIn, *ImgOut, *PlanR, *PlanV, *PlanB;

	allocation_tableau(ImgIn, OCTET, nTaille3);
	lire_image_ppm(cNomImgLue, ImgIn, nH * nW);

	allocation_tableau(PlanR, OCTET, nTaille);
	planR(PlanR, ImgIn, nTaille);

	allocation_tableau(PlanV, OCTET, nTaille);
	planV(PlanV, ImgIn, nTaille);

	allocation_tableau(PlanB, OCTET, nTaille);
	planB(PlanB, ImgIn, nTaille);

	flouter_image(PlanR, nH, nW, rayon);
	flouter_image(PlanV, nH, nW, rayon);
	flouter_image(PlanB, nH, nW, rayon);

	allocation_tableau(ImgOut, OCTET, nTaille3);
	for (int i = 0; i < nTaille; i++) {
		ImgOut[i * 3] = PlanR[i];
		ImgOut[i * 3 + 1] = PlanV[i];
		ImgOut[i * 3 + 2] = PlanB[i];
	}

	ecrire_image_ppm(cNomImgEcrite, ImgOut, nH, nW);

	free(ImgIn);
	free(ImgOut);
	free(PlanR);
	free(PlanV);
	free(PlanB);
}

/*============================== TP 4 =============================================*/

void norme_gradient_pgm(char cNomImgLue[250], char cNomImgEcrite[250], char* filtreMoyenne, char* filtreGauss) { // peut être ajouter rayon en param ?
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	int masqueHorizontal[] = {-1, 0, 1,
				  -2, 0, 2,
				  -1, 0, 1};

	int masqueVertical[] = {-1, -2, 1,
				0, 0, 0,
				1, 2, 1};

	// appliquer filtre moyenneur
	if (strcmp(filtreMoyenne, "V") == 0) {
		filtreMoyenneurPGM(ImgIn, nW, nH, 1);
	}
	// appliquer filtre gaussien
	if (strcmp(filtreMoyenne, "V") == 0) {
		filtreGaussienPGM(ImgIn, nW, nH, 1);
	}

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			// ImgOut[i*nW + j] = getNormeGradient(ImgIn, nW, nH, i, j, 1, masqueHorizontal, masqueVertical);
			int diffY = 0;
			int diffX = 0;

			if (j < nW - 1) {
				diffX = ImgIn[i * nW + j + 1] - ImgIn[i * nW + j];
			}
			if (i < nH - 1) {
				diffY = ImgIn[(i + 1) * nW + j] - ImgIn[i * nW + j];
			}
			int res = sqrt(pow(diffX, 2) + pow(diffY, 2));
			ImgOut[i * nW + j] = clamp(res);
		}
	}

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void seuil_hysteresis_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int seuilBas, int seuilHaut) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut, *ImgTemp;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nTaille);
	allocation_tableau(ImgTemp, OCTET, nTaille);
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			if (ImgIn[i * nW + j] <= seuilBas) {
				ImgTemp[i * nW + j] = 0;
			} else if (ImgIn[i * nW + j] >= seuilHaut) {
				ImgTemp[i * nW + j] = 255;
			} else {
				ImgTemp[i * nW + j] = 127;
			}
		}
	}

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = checkVoisinsHysteresisV2(ImgTemp, nW, nH, i, j, 1, seuilBas, seuilHaut);
		}
	}

	ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

/*============================== TP 5 =============================================*/

void RGBtoY(char cNomImgPPM[250], char cNomImgPGM[250]) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_ppm(cNomImgPPM, &nH, &nW);
	nTaille = nH * nW * 3;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_ppm(cNomImgPPM, ImgIn, nH * nW);

	allocation_tableau(ImgOut, OCTET, nH * nW); // Pour l'image en niveaux de gris

	for (int i = 0; i < nH * nW; i++) {
		int r = ImgIn[3 * i];
		int g = ImgIn[3 * i + 1];
		int b = ImgIn[3 * i + 2];
		ImgOut[i] = (OCTET)(0.299 * r + 0.587 * g + 0.114 * b);
	}

	ecrire_image_pgm(cNomImgPGM, ImgOut, nH, nW);

	free(ImgIn);
	free(ImgOut);
}

double calculerEQM(char cNomImgPGM1[250], char cNomImgPGM2[250]) {
	int nH, nW, nTaille;
	OCTET *Img1, *Img2;
	double eqm = 0.0;

	lire_nb_lignes_colonnes_image_pgm(cNomImgPGM1, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(Img1, OCTET, nTaille);
	lire_image_pgm(cNomImgPGM1, Img1, nTaille);

	allocation_tableau(Img2, OCTET, nTaille);
	lire_image_pgm(cNomImgPGM2, Img2, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			eqm += pow((double)Img1[i * nW + j] - (double)Img2[i * nW + j], 2);
		}
	}
	eqm /= nTaille;

	free(Img1);
	free(Img2);

	return eqm;
}

void RGBtoYCbCr(char cNomImgPPM[250]) {
	int nH, nW, nTaille;
	OCTET *ImgIn, *ImgY, *ImgCb, *ImgCr;

	char cNomImgY[250], cNomImgCb[250], cNomImgCr[250];
	char* dot;
	dot = strrchr(cNomImgPPM, '.');
	int index = dot - cNomImgPPM;

	// Copier la base du nom sans l'extension pour Y
	strncpy(cNomImgY, cNomImgPPM, index);
	cNomImgY[index] = '\0'; // S'assurer que la chaîne est terminée correctement
	strcat(cNomImgY, "_Y.pgm");

	// Répéter le processus pour Cb et Cr
	strncpy(cNomImgCb, cNomImgPPM, index);
	cNomImgCb[index] = '\0';
	strcat(cNomImgCb, "_Cb.pgm");

	strncpy(cNomImgCr, cNomImgPPM, index);
	cNomImgCr[index] = '\0';
	strcat(cNomImgCr, "_Cr.pgm");

	// Lecture des dimensions de l'image PPM
	lire_nb_lignes_colonnes_image_ppm(cNomImgPPM, &nH, &nW);
	nTaille = nH * nW * 3; // Taille pour l'image en couleur

	// Allocation et lecture de l'image PPM
	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_ppm(cNomImgPPM, ImgIn, nH * nW);

	// Allocation pour les images Y, Cb, et Cr
	allocation_tableau(ImgY, OCTET, nH * nW);
	allocation_tableau(ImgCb, OCTET, nH * nW);
	allocation_tableau(ImgCr, OCTET, nH * nW);

	for (int i = 0; i < nH * nW; i++) {
		int r = ImgIn[3 * i];
		int g = ImgIn[3 * i + 1];
		int b = ImgIn[3 * i + 2];

		// Calcul des composantes Y, Cb, Cr
		ImgY[i] = (OCTET)(0.299 * r + 0.587 * g + 0.114 * b);
		ImgCb[i] = (OCTET)(-0.168736 * r - 0.331264 * g + 0.5 * b + 128);
		ImgCr[i] = (OCTET)(0.5 * r - 0.418688 * g - 0.081312 * b + 128);
	}

	// Écriture des images Y, Cb, Cr au format PGM
	ecrire_image_pgm(cNomImgY, ImgY, nH, nW);
	ecrire_image_pgm(cNomImgCb, ImgCb, nH, nW);
	ecrire_image_pgm(cNomImgCr, ImgCr, nH, nW);

	// Libération de la mémoire
	free(ImgIn);
	free(ImgY);
	free(ImgCb);
	free(ImgCr);
}

void YCbCrtoRGB(char cNomImgY[250], char cNomImgCb[250], char cNomImgCr[250], char cNomImgPPM[250]) {
	int nH, nW, nTaille;
	OCTET *ImgY, *ImgCb, *ImgCr, *ImgPPM;

	// Lecture des dimensions de l'une des images Y, Cb, ou Cr (elles ont toutes les mêmes dimensions)
	lire_nb_lignes_colonnes_image_pgm(cNomImgY, &nH, &nW);
	nTaille = nH * nW;

	// Allocation et lecture des images Y, Cb, et Cr
	allocation_tableau(ImgY, OCTET, nTaille);
	lire_image_pgm(cNomImgY, ImgY, nTaille);

	allocation_tableau(ImgCb, OCTET, nTaille);
	lire_image_pgm(cNomImgCb, ImgCb, nTaille);

	allocation_tableau(ImgCr, OCTET, nTaille);
	lire_image_pgm(cNomImgCr, ImgCr, nTaille);

	// Allocation pour l'image PPM de sortie
	allocation_tableau(ImgPPM, OCTET, 3 * nTaille);

	for (int i = 0; i < nTaille; i++) {
		double Y = ImgY[i];
		double Cb = ImgCb[i];
		double Cr = ImgCr[i];

		double R = Y + 1.402 * (Cr - 128);
		double G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128);
		double B = Y + 1.772 * (Cb - 128);

		// Ajustement des valeurs pour qu'elles soient dans l'intervalle [0, 255] et écriture dans l'image PPM
		ImgPPM[3 * i] = clamp(R);
		ImgPPM[3 * i + 1] = clamp(G);
		ImgPPM[3 * i + 2] = clamp(B);
	}

	// Écriture de l'image PPM
	ecrire_image_ppm(cNomImgPPM, ImgPPM, nH, nW);

	// Libération de la mémoire
	free(ImgY);
	free(ImgCb);
	free(ImgCr);
	free(ImgPPM);
}

void modifY(char cNomImgY[250], int k, char cNomImgOut[250]) {
	int nH, nW, nTaille;
	OCTET *Img1, *Img2;

	lire_nb_lignes_colonnes_image_pgm(cNomImgY, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(Img1, OCTET, nTaille);
	lire_image_pgm(cNomImgY, Img1, nTaille);

	allocation_tableau(Img2, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			Img2[i * nW + j] = clamp(Img1[i * nW + j] + k);
		}
	}

	ecrire_image_pgm(cNomImgOut, Img2, nH, nW);

	free(Img1);
	free(Img2);
}


#endif // IMAGE_H