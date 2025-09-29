#include "AES.hpp"
#include "../src/image_ppm.h"
#include "../src/codage.h"

#include <iostream>

void encryptAES(char cNomImgLue[250], char cNomImgEcrite[250], int tailleBloc){
    int nH, nW, nTaille;

	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
	allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };


    Cipher::Aes<128> aes(key);

    for (int i = 0; i < nTaille; i += tailleBloc) {
        unsigned char block[tailleBloc] = {0}; 
        for(int j = 0; j < tailleBloc; j++){
            block[j] = ImgIn[i + j];
        }
        aes.encrypt_block(block); 
        for(int j = i ; j < i + tailleBloc ; ++j){
            ImgOut[j] = block[j-i];
        }        
    }


    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void encryptCBC(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit){
    int nH, nW, nTaille;

	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
	allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };


    int tailleBloc{static_cast<int>(vecInit.size())};
    
    Cipher::Aes<128> aes(key);

    //faire xor entre 1er bloc et vect
    unsigned char block[tailleBloc] = {0};
    for(int j = 0; j < tailleBloc; j++){
        block[j] = vecInit[j] ^ ImgIn[j];
    }

    // chiffrer le resultat 
    aes.encrypt_block(block); 

    // ecrire dans l'image
    for(int j = 0; j < tailleBloc; j++){
        ImgOut[j] = block[j];
    }


    for (int i = tailleBloc; i < nTaille; i += tailleBloc) {
        unsigned char blockCurr[tailleBloc] = {0}; 
        for(int j = 0; j < tailleBloc; j++){
            blockCurr[j] = ImgIn[i + j] ^ ImgOut[i + j - tailleBloc];
        }
        aes.encrypt_block(blockCurr); 
        for(int j = i ; j < i + tailleBloc ; ++j){
            ImgOut[j] = blockCurr[j-i];
        }        
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}

void encryptCFB(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit){
    int nH, nW, nTaille;

	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
	allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };


    int tailleBloc{static_cast<int>(vecInit.size())};
    
    Cipher::Aes<128> aes(key);

    //chiffre aes le vecteur init
    unsigned char chiffre[tailleBloc] = {0};
    for(int j = 0; j < tailleBloc; j++){
        chiffre[j] = vecInit[j];
    }
    aes.encrypt_block(chiffre); 

    // recup le bloc clair et ou exclusif avec chiffre precedent
    unsigned char block[tailleBloc] = {0}; 
    for(int j = 0; j < tailleBloc; j++){
        block[j] = ImgIn[0 + j] ^ chiffre[j];
        ImgOut[j] =  block[j] ;
    }
     
    // pour le reste je recup les taillebloc de imgout precedent
    // j'encrype aes avec la clé 
    // le je fais avec le encrypte un xor 
    // j'écrit dans imgout
    for (int i = tailleBloc; i < nTaille; i += tailleBloc) {
        unsigned char blockCurr[tailleBloc]  = {0}; 
        for(int j = 0; j < tailleBloc; j++){
            blockCurr[j] = ImgOut[i - tailleBloc + j];
        }
        aes.encrypt_block(blockCurr); 
        for(int j = 0; j < tailleBloc; j++){
            ImgOut[j + i]  = blockCurr[j] ^ ImgIn[i + j];
        }
    }


    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}


void encryptOFB(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit){
    int nH, nW, nTaille;

	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
	allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };


    int tailleBloc{static_cast<int>(vecInit.size())};
    
    Cipher::Aes<128> aes(key);

    //chiffre aes le vecteur init
    unsigned char chiffre[tailleBloc] = {0};
    for(int j = 0; j < tailleBloc; j++){
        chiffre[j] = vecInit[j];
    }
    aes.encrypt_block(chiffre); 

    // recup le bloc clair et ou exclusif avec chiffre precedent
    unsigned char block[tailleBloc] = {0}; 
    for(int j = 0; j < tailleBloc; j++){
        block[j] = ImgIn[0 + j] ^ chiffre[j];
        ImgOut[j] =  block[j] ;
    }
     
    // pour le reste je recup les taillebloc de imgout precedent
    // j'encrype aes avec la clé 
    // le je fais avec le encrypte un xor 
    // j'écrit dans imgout
    for (int i = tailleBloc; i < nTaille; i += tailleBloc) {
        unsigned char blockCurr[tailleBloc]  = {0}; 
        for(int j = 0; j < tailleBloc; j++){
            blockCurr[j] = chiffre[j];
        }
        aes.encrypt_block(blockCurr); 

        // je garde le xor 
        for(int j = 0; j < tailleBloc; j++){
            chiffre[j] = blockCurr[j];
        }

        // j'ecrit dans l'image le xor
        for(int j = 0; j < tailleBloc; j++){
            ImgOut[j + i]  = blockCurr[j] ^ ImgIn[i + j];
        }
    }


    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
	free(ImgIn);
	free(ImgOut);
}


void encryptCTR(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit){
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
    allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };

    int tailleBloc{static_cast<int>(vecInit.size())};
    Cipher::Aes<128> aes(key);

    // initialiser compteur (plain)
    unsigned char counter[16] = {0};
    for (int k = 0; k < 16 && k < (int)vecInit.size(); ++k) counter[k] = vecInit[k];

    // chiffrer bloc par bloc : chiffrer une copie du compteur (keystream), puis incrémenter le compteur en clair
    for (int i = 0; i < nTaille; i += 16) {
        unsigned char keystream[16];
        // copier compteur dans keystream
        for (int k = 0; k < 16; ++k) keystream[k] = counter[k];

        // chiffrer la copie -> keystream
        aes.encrypt_block(keystream);

        // XOR avec les données
        for (int j = 0; j < 16 && (i + j) < nTaille; ++j) {
            ImgOut[i + j] = ImgIn[i + j] ^ keystream[j];
        }

        // incrémenter le compteur (big-endian)
        for (int k = 15; k >= 0; --k) {
            if (++counter[k] != 0) break;
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
    free(ImgIn);
    free(ImgOut);
}
































void decryptAES(char cNomImgLue[250], char cNomImgEcrite[250], int tailleBloc) {
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
    allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };

    Cipher::Aes<128> aes(key);

    for (int i = 0; i < nTaille; i += tailleBloc) {
        unsigned char block[tailleBloc] = {0};
        for (int j = 0; j < tailleBloc; j++) {
            block[j] = ImgIn[i + j];
        }
        aes.decrypt_block(block);
        for (int j = i; j < i + tailleBloc; ++j) {
            ImgOut[j] = block[j - i];
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
    free(ImgIn);
    free(ImgOut);
}

void decryptCBC(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit) {
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);
    allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };

    int tailleBloc = static_cast<int>(vecInit.size());
    Cipher::Aes<128> aes(key);

    // prevCipher = IV pour le premier bloc, puis devient le bloc chiffré précédent
    std::vector<unsigned char> prevCipher(tailleBloc);
    for (int j = 0; j < tailleBloc; ++j) prevCipher[j] = vecInit[j];

    for (int i = 0; i < nTaille; i += tailleBloc) {
        // copier le bloc chiffré courant
        unsigned char currCipher[256]; // taille suffisante (tailleBloc <= 256)
        for (int j = 0; j < tailleBloc; ++j) currCipher[j] = ImgIn[i + j];

        // décrypter le bloc courant dans place dans 'plain'
        unsigned char plain[256];
        for (int j = 0; j < tailleBloc; ++j) plain[j] = currCipher[j];
        aes.decrypt_block(plain);

        // XOR avec prevCipher (IV pour premier bloc)
        for (int j = 0; j < tailleBloc; ++j) {
            ImgOut[i + j] = plain[j] ^ prevCipher[j];
        }

        // mettre à jour prevCipher = currCipher (bloc chiffré)
        for (int j = 0; j < tailleBloc; ++j) prevCipher[j] = currCipher[j];
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
    free(ImgIn);
    free(ImgOut);
}

void decryptOFB(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit) {
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);
    allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };

    int tailleBloc = static_cast<int>(vecInit.size());
    Cipher::Aes<128> aes(key);

    unsigned char chiffre[tailleBloc] = {0};
    for (int j = 0; j < tailleBloc; j++) {
        chiffre[j] = vecInit[j];
    }

    // Déchiffrer l'image bloc par bloc
    for (int i = 0; i < nTaille; i += tailleBloc) {
        unsigned char blockCurr[tailleBloc] = {0};
        aes.encrypt_block(chiffre); // Générer le flux chiffré
        for (int j = 0; j < tailleBloc; j++) {
            ImgOut[i + j] = ImgIn[i + j] ^ chiffre[j];
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
    free(ImgIn);
    free(ImgOut);
}

void decryptCTR(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit) {
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);
    allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };

    Cipher::Aes<128> aes(key);

    // initialiser compteur (plain)
    unsigned char counter[16] = {0};
    for (int k = 0; k < 16 && k < (int)vecInit.size(); ++k) counter[k] = vecInit[k];

    // déchiffrement CTR == chiffrement : chiffrer une copie du compteur, XOR, incrémenter le compteur en clair
    for (int i = 0; i < nTaille; i += 16) {
        unsigned char keystream[16];
        for (int k = 0; k < 16; ++k) keystream[k] = counter[k];

        aes.encrypt_block(keystream);

        for (int j = 0; j < 16 && (i + j) < nTaille; ++j) {
            ImgOut[i + j] = ImgIn[i + j] ^ keystream[j];
        }

        for (int k = 15; k >= 0; --k) {
            if (++counter[k] != 0) break;
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
    free(ImgIn);
    free(ImgOut);
}

void decryptCFB(char cNomImgLue[250], char cNomImgEcrite[250], std::vector<unsigned char> vecInit) {
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);
    allocation_tableau(ImgOut, OCTET, nTaille);

    unsigned char key[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };

    int tailleBloc = static_cast<int>(vecInit.size());
    Cipher::Aes<128> aes(key);

    // Initialiser le vecteur d'initialisation
    unsigned char chiffre[tailleBloc] = {0};
    for (int j = 0; j < tailleBloc; j++) {
        chiffre[j] = vecInit[j];
    }

    // Déchiffrer le premier bloc
    aes.encrypt_block(chiffre); // Chiffrer le vecteur d'initialisation
    for (int j = 0; j < tailleBloc; j++) {
        ImgOut[j] = ImgIn[j] ^ chiffre[j];
    }

    // Déchiffrer les blocs suivants
    for (int i = tailleBloc; i < nTaille; i += tailleBloc) {
        unsigned char blockCurr[tailleBloc] = {0};

        // Copier le bloc chiffré précédent dans blockCurr
        for (int j = 0; j < tailleBloc; j++) {
            blockCurr[j] = ImgIn[i - tailleBloc + j];
        }

        // Chiffrer le bloc précédent
        aes.encrypt_block(blockCurr);

        // XOR avec le flux chiffré pour récupérer les données originales
        for (int j = 0; j < tailleBloc; j++) {
            ImgOut[i + j] = ImgIn[i + j] ^ blockCurr[j];
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
    free(ImgIn);
    free(ImgOut);
}