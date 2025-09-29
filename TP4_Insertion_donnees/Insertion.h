#include "../src/image_ppm.h"
#include "../src/codage.h"

#include <algorithm>
#include <stdlib.h>


OCTET getValue(OCTET in, int powerof2) {
    return std::clamp(((in >> powerof2) & 1) * 255, 0 ,255); 
}

OCTET replace(OCTET in, int powerof2, int value) {
    OCTET mask = 1 << powerof2;
    if(value > 0){
        return in | mask;
    }else {
        return in & ~mask; 
    }
}


void extractBinaryPlans(char cNomImgLue[250]){
    int nH, nW, nTaille;
	OCTET *ImgIn, *Img0, *Img1,*Img2,*Img3,*Img4,*Img5,*Img6,*Img7;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

	allocation_tableau(ImgIn, OCTET, nTaille);
    allocation_tableau(Img0, OCTET, nTaille);
    allocation_tableau(Img1, OCTET, nTaille);
    allocation_tableau(Img2, OCTET, nTaille);
    allocation_tableau(Img3, OCTET, nTaille);
    allocation_tableau(Img4, OCTET, nTaille);
    allocation_tableau(Img5, OCTET, nTaille);
    allocation_tableau(Img6, OCTET, nTaille);
    allocation_tableau(Img7, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);


    for(int i {0}; i < nTaille; ++i){
        Img0[i] = getValue(ImgIn[i], 0);
        Img1[i] = getValue(ImgIn[i], 1);
        Img2[i] = getValue(ImgIn[i], 2);
        Img3[i] = getValue(ImgIn[i], 3);
        Img4[i] = getValue(ImgIn[i], 4);
        Img5[i] = getValue(ImgIn[i], 5);
        Img6[i] = getValue(ImgIn[i], 6);
        Img7[i] = getValue(ImgIn[i], 7);
    }

    char nameBuffer[300];
    char* names[8];

    for (int i = 0; i < 8; ++i) {
        sprintf(nameBuffer, "img_%d.pgm", i);
        names[i] = (char*)malloc(strlen(nameBuffer) + 1);
        strcpy(names[i], nameBuffer);
    }

    ecrire_image_pgm(names[0], Img0, nH, nW);
    ecrire_image_pgm(names[1], Img1, nH, nW);
    ecrire_image_pgm(names[2], Img2, nH, nW);
    ecrire_image_pgm(names[3], Img3, nH, nW);
    ecrire_image_pgm(names[4], Img4, nH, nW);
    ecrire_image_pgm(names[5], Img5, nH, nW);
    ecrire_image_pgm(names[6], Img6, nH, nW);
    ecrire_image_pgm(names[7], Img7, nH, nW);

    for (int i = 0; i < 8; ++i) {
        free((void*)names[i]);
    }


    free(ImgIn);
    free(Img0);
    free(Img1);
    free(Img2);
    free(Img3);
    free(Img4);
    free(Img5);
    free(Img6);
    free(Img7);
}



std::vector<OCTET> generateMsg(char cNomImgLue[250]){
    int nH, nW, nTaille;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;

    srand(25);

    std::vector<OCTET> msg(nTaille);
    for(int i {0}; i < nTaille; ++i){
        msg[i] = rand()%2;
    }
    return msg;
}


void addMsgToImage(char cNomImgLue[250], char cOutImg[250], int bitToReplace){
    int nH, nW, nTaille;

	OCTET *ImgIn, *ImgOut;

	lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;
    allocation_tableau(ImgIn, OCTET, nTaille);
    allocation_tableau(ImgOut, OCTET, nTaille);
	lire_image_pgm(cNomImgLue, ImgIn, nH * nW);


    std::vector<OCTET> msg = generateMsg(cNomImgLue);

    for(int i{0}; i < nTaille; ++i){
        ImgOut[i] = replace(ImgIn[i], bitToReplace, msg[i]);
    }

    ecrire_image_pgm(cOutImg, ImgOut, nH, nW);

    free(ImgIn);
    free(ImgOut);

}