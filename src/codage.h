#ifndef CODAGE_h
#define CODAGE_h





#include <cstdlib>
#include <math.h>
#include <vector>
#include <iostream>
#include <sys/stat.h>  // Pour struct stat
#include <unistd.h>    // Pour stat()



// c1 : 9;68;84
// c2 : 253;186;12

typedef unsigned char OCTET;

class Vec3{
    public: 
        float r,v,b;

        Vec3(float a, float b, float c){
            r = a;
            v = b;
            this->b = c;
        }
        
        Vec3() : r(0), v(0), b(0) {}

    Vec3& operator/=(const float& a) {
        if (a != 0) {  
            r /= a;
            v /= a;
            b /= a;
        }
        return *this;
    }


    Vec3& operator+=(const Vec3& a) {  
        r += a.r;
        v += a.v;
        b += a.b;
        return *this;
    }

    Vec3& operator-=(const Vec3& a) {  
        r -= a.r;
        v -= a.v;
        b -= a.b;
        return *this;
    }


    public:
        static float dist(const Vec3& a, const Vec3& b) {  // Utilisation de const Vec3&
            return sqrt(pow((b.r - a.r), 2) + pow((b.v - a.v), 2) + pow((b.b - a.b), 2));
        }

};


void k_mean_recursive(std::vector<Vec3>& grp1, std::vector<Vec3>& grp2, Vec3& c1, Vec3& c2, int depth, int depthMax) {
    if (depth == depthMax) {
        return;
    }

    Vec3 newC1(0, 0, 0);
    for (const Vec3& color : grp1) {
        newC1 += color;
    }
    if (!grp1.empty()) {
        newC1 /= grp1.size();
    }


    Vec3 newC2(0, 0, 0);
    for (const Vec3& color : grp2) {
        newC2 += color;
    }
    if (!grp2.empty()) {
        newC2 /= grp2.size();
    }


    std::vector<Vec3> new_grp1;
    std::vector<Vec3> new_grp2;


    for (const Vec3& color : grp1) {
        float distToC1 = Vec3::dist(color, newC1);
        float distToC2 = Vec3::dist(color, newC2);
        if (distToC1 < distToC2) {
            new_grp1.push_back(color);
        } else {
            new_grp2.push_back(color);
        }
    }

    for (const Vec3& color : grp2) {
        float distToC1 = Vec3::dist(color, newC1);
        float distToC2 = Vec3::dist(color, newC2);
        if (distToC1 < distToC2) {
            new_grp1.push_back(color);
        } else {
            new_grp2.push_back(color);
        }
    }


    grp1 = new_grp1;
    grp2 = new_grp2;
    c1 = newC1;
    c2 = newC2;

    k_mean_recursive(grp1, grp2, c1, c2, depth + 1, depthMax);
}


void k_mean(char cNomImgLue[250], char cNomImgEcrite[250], char cNomImgEcrite2[250]){
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut, *ImgOut2;
    
    lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW * 3;
  
    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_ppm(cNomImgLue, ImgIn, nH * nW);
    allocation_tableau(ImgOut, OCTET, nTaille);
    allocation_tableau(ImgOut2, OCTET, nTaille);

    std::vector<Vec3> grp1;
    std::vector<Vec3> grp2;

    Vec3 c1 (9, 68, 84);
    Vec3 c2 (253, 186, 12);

    Vec3 baseC1 (9, 68, 84);
    Vec3 baseC2 (253, 186, 12);
    
    for (int i = 0; i < nTaille; i += 3) {
        Vec3 color(ImgIn[i], ImgIn[i + 1], ImgIn[i + 2]);
        float distToC1 = Vec3::dist(color, c1);
        float distToC2 = Vec3::dist(color, c2);

        if (distToC1 < distToC2) {
            grp1.push_back(color);
        } else {
            grp2.push_back(color);
        }
    }


    k_mean_recursive(grp1, grp2, c1, c2, 0, 20);

    for (int i = 0; i < nTaille; i += 3) {
        Vec3 color(ImgIn[i], ImgIn[i + 1], ImgIn[i + 2]);
        float distToC1 = Vec3::dist(color, baseC1);
        float distToC2 = Vec3::dist(color, baseC2);

        if (distToC1 < distToC2) {
            ImgOut[i] = static_cast<int>(baseC1.r);
            ImgOut[i + 1] = static_cast<int>(baseC1.v);
            ImgOut[i + 2] = static_cast<int>(baseC1.b);
        } else {
            ImgOut[i] = static_cast<int>(baseC2.r);
            ImgOut[i + 1] = static_cast<int>(baseC2.v);
            ImgOut[i + 2] = static_cast<int>(baseC2.b);
        }
    }

    for (int i = 0; i < nTaille; i += 3) {
        Vec3 color(ImgIn[i], ImgIn[i + 1], ImgIn[i + 2]);
        float distToC1 = Vec3::dist(color, c1);
        float distToC2 = Vec3::dist(color, c2);

        if (distToC1 < distToC2) {
            ImgOut2[i] = static_cast<int>(c1.r);
            ImgOut2[i + 1] = static_cast<int>(c1.v);
            ImgOut2[i + 2] = static_cast<int>(c1.b);
        } else {
            ImgOut2[i] = static_cast<int>(c2.r);
            ImgOut2[i + 1] = static_cast<int>(c2.v);
            ImgOut2[i + 2] = static_cast<int>(c2.b);
        }
    }

    ecrire_image_ppm(cNomImgEcrite, ImgOut, nH, nW);
    ecrire_image_ppm(cNomImgEcrite2, ImgOut2, nH, nW);

}

void k_mean_256(char cNomImgLue[250], char cNomImgEcrite[250]) {
    int nH, nW, nTaille;

    OCTET *ImgIn, *ImgOut;
    
    lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW * 3;
  
    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_ppm(cNomImgLue, ImgIn, nH * nW);
    allocation_tableau(ImgOut, OCTET, nTaille);

    std::vector<Vec3> centroids(256);
    for (int i = 0; i < 256; i++) {
        int randomIndex = rand() % (nTaille / 3) * 3;
        centroids[i] = Vec3(ImgIn[randomIndex], ImgIn[randomIndex + 1], ImgIn[randomIndex + 2]);
    }

    for (int iteration = 0; iteration < 12; iteration++) {
        std::vector<std::vector<Vec3>> groups(256);

        for (int i = 0; i < nTaille; i += 3) {
            Vec3 color(ImgIn[i], ImgIn[i + 1], ImgIn[i + 2]);
            int bestCentroid = 0;
            float minDist = Vec3::dist(color, centroids[0]);

            for (int c = 1; c < 256; c++) {
                float dist = Vec3::dist(color, centroids[c]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCentroid = c;
                }
            }
            groups[bestCentroid].push_back(color);
        }

        for (int c = 0; c < 256; c++) {
            Vec3 newCentroid(0, 0, 0);
            for (const Vec3& color : groups[c]) {
                newCentroid += color;
            }
            if (!groups[c].empty()) {
                newCentroid /= groups[c].size();
            }
            centroids[c] = newCentroid;
        }
    }

    for (int i = 0; i < nTaille; i += 3) {
        Vec3 color(ImgIn[i], ImgIn[i + 1], ImgIn[i + 2]);
        int bestCentroid = 0;
        float minDist = Vec3::dist(color, centroids[0]);

        for (int c = 1; c < 256; c++) {
            float dist = Vec3::dist(color, centroids[c]);
            if (dist < minDist) {
                minDist = dist;
                bestCentroid = c;
            }
        }

        ImgOut[i] = static_cast<int>(centroids[bestCentroid].r);
        ImgOut[i + 1] = static_cast<int>(centroids[bestCentroid].v);
        ImgOut[i + 2] = static_cast<int>(centroids[bestCentroid].b);
    }

    ecrire_image_ppm(cNomImgEcrite, ImgOut, nH, nW);
}

void PSNR_PPM (char cNomImgLue[250], char cNomImgLue2[250]){
    int nH, nW, nTaille;

    OCTET *ImgOriginal, *ImgModifiee;
    
    lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW * 3;
  
    allocation_tableau(ImgOriginal, OCTET, nTaille);
    lire_image_ppm(cNomImgLue, ImgOriginal, nH * nW);

    allocation_tableau(ImgModifiee, OCTET, nTaille);
    lire_image_ppm(cNomImgLue2, ImgModifiee, nH * nW);


    float eqm = 0.0;

    for (int i = 0; i < nTaille; i += 3) {
        int diffR = ImgOriginal[i] - ImgModifiee[i];
        int diffG = ImgOriginal[i + 1] - ImgModifiee[i + 1];
        int diffB = ImgOriginal[i + 2] - ImgModifiee[i + 2];

        eqm += diffR * diffR + diffG * diffG + diffB * diffB;
    }

    eqm /= nTaille; 
    std::cout << " EQM : " << eqm << std::endl;
    
    float pnsr = 10* log10(pow(255,2) / eqm);
    std::cout << " PNSR : " << pnsr  << " dB " << std::endl;


}

void PSNR_PGM (char cNomImgLue[250], char cNomImgLue2[250]){
    int nH, nW, nTaille;

    OCTET *ImgOriginal, *ImgModifiee;
    
    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;
  
    allocation_tableau(ImgOriginal, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgOriginal, nH * nW);

    allocation_tableau(ImgModifiee, OCTET, nTaille);
    lire_image_pgm(cNomImgLue2, ImgModifiee, nH * nW);


    float eqm = 0.0;

    for (int i = 0; i < nTaille; i += 3) {
        int diff = ImgOriginal[i] - ImgModifiee[i];
        eqm += diff * diff;
    }

    eqm /= nTaille; 
    std::cout << " EQM : " << eqm << std::endl;
    
    float pnsr = 10* log10(pow(255,2) / eqm);
    std::cout << " PNSR : " << pnsr  << " dB " << std::endl;

}


void reduce_image_pgm(OCTET *ImgIn, OCTET *ImgOut, int nH, int nW, int reduction) {
    int nH2 = nH / reduction; 
    int nW2 = nW / reduction;

    for (int i = 0; i < nH2; i++) {
        for (int j = 0; j < nW2; j++) {
            ImgOut[i * nW2 + j] = ImgIn[(i * reduction) * nW + (j * reduction)];
        }
    }
}


float interpolate(float q11, float q12, float q21, float q22, float x, float y) {
    return q11 * (1 - x) * (1 - y) +
           q12 * x * (1 - y) +
           q21 * (1 - x) * y +
           q22 * x * y;
}

void resample_image_pgm(OCTET *ImgIn, OCTET *ImgOut, int nH, int nW, int nH2, int nW2) {
    for (int i = 0; i < nH2; i++) {
        for (int j = 0; j < nW2; j++) {
            // Calcul des coordonnées dans l'image d'entrée
            float x = (float)j * (nW - 1) / (nW2 - 1); // Position x dans l'image d'entrée
            float y = (float)i * (nH - 1) / (nH2 - 1); // Position y dans l'image d'entrée

            int x1 = (int)x; // Coordonnée entière de gauche
            int y1 = (int)y; // Coordonnée entière du haut
            int x2 = (x1 + 1 < nW) ? x1 + 1 : x1; // Coordonnée entière de droite (limité au bord droit)
            int y2 = (y1 + 1 < nH) ? y1 + 1 : y1; // Coordonnée entière du bas (limité au bord bas)

            // Valeurs des 4 pixels environnants (x1, y1), (x2, y1), (x1, y2), (x2, y2)
            float q11 = ImgIn[y1 * nW + x1];
            float q12 = ImgIn[y1 * nW + x2];
            float q21 = ImgIn[y2 * nW + x1];
            float q22 = ImgIn[y2 * nW + x2];

            // Calcul des fractions pour l'interpolation
            float x_frac = x - x1; // Fraction de la distance dans la direction x
            float y_frac = y - y1; // Fraction de la distance dans la direction y

            // Interpolation bilinéaire
            ImgOut[i * nW2 + j] = (OCTET)(
                q11 * (1 - x_frac) * (1 - y_frac) + 
                q12 * x_frac * (1 - y_frac) + 
                q21 * (1 - x_frac) * y_frac + 
                q22 * x_frac * y_frac
            );
        }
    }
}




void ex3_tp2(char cNomImgLue[250], char cNomImgSortie[250]){

    int nH, nW, nTaille;
    int nTaille3;
    
    OCTET *ImgIn, *ImgOut, *PlanR, *PlanV, *PlanB;

    lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;
    nTaille3 = nTaille * 3;

    allocation_tableau(ImgIn, OCTET, nTaille3);
    lire_image_ppm(cNomImgLue, ImgIn, nTaille);

    allocation_tableau(PlanR, OCTET, nTaille);
    planR(PlanR, ImgIn, nTaille);

    allocation_tableau(PlanV, OCTET, nTaille);
    planV(PlanV, ImgIn, nTaille);

    allocation_tableau(PlanB, OCTET, nTaille);
    planB(PlanB, ImgIn, nTaille);

    char R[] = "img_R.pgm";
    char G[] = "img_G.pgm";
    char B[] = "img_B.pgm";

    ecrire_image_pgm(R, PlanR, nH, nW);
    ecrire_image_pgm(G, PlanV, nH, nW);
    ecrire_image_pgm(B, PlanB, nH, nW);

    OCTET *ImgR_reduite, *ImgB_reduite;
    allocation_tableau(ImgR_reduite, OCTET, nTaille/2);
    allocation_tableau(ImgB_reduite, OCTET, nTaille/2);

    reduce_image_pgm(PlanR, ImgR_reduite, nH, nW, 2);
    reduce_image_pgm(PlanB, ImgB_reduite, nH, nW, 2);

    OCTET *ImgR_re_sampled, *ImgB_re_sampled;
    allocation_tableau(ImgR_re_sampled, OCTET, nTaille);
    allocation_tableau(ImgB_re_sampled, OCTET, nTaille);


    resample_image_pgm(ImgR_reduite, ImgR_re_sampled, nH/2 , nW/2, nH, nW);
    resample_image_pgm( ImgB_reduite, ImgB_re_sampled, nH/2 , nW/2, nH, nW);

    char RR[] = "R_resampled.pgm";
    char GR[] = "B_resampled.pgm";

    ecrire_image_pgm(RR, ImgR_re_sampled, nH, nW);
    ecrire_image_pgm(GR, ImgB_re_sampled, nH, nW);

    allocation_tableau(ImgOut, OCTET, nTaille3);
    for (int i = 0; i < nTaille; i++) {
        ImgOut[i*3] = ImgR_re_sampled[i];
        ImgOut[i*3+1] = PlanV[i];
        ImgOut[i*3+2] = ImgB_re_sampled[i];
    }

    ecrire_image_ppm(cNomImgSortie, ImgOut, nH, nW);

    free(ImgR_reduite);
    free(ImgB_reduite);

    free(ImgR_re_sampled);
    free(ImgB_re_sampled);

    free(ImgIn);
    free(ImgOut);

    free(PlanR);
    free(PlanV);
    free(PlanB);


}


void re_sample_ycbcr(char cNomImgLue[250], char cNomImgLue2[250], char cNomImgLue3[250]){

    int nH, nW, nTaille;
    
    OCTET *ImgY, *ImgCb, * ImgCr;

    OCTET *ImgYOut, *ImgCbOut, * ImgCrOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    // lectures de images

    allocation_tableau(ImgY, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgY, nTaille);

    allocation_tableau(ImgCb, OCTET, nTaille);
    lire_image_pgm(cNomImgLue2, ImgCb, nTaille);

    allocation_tableau(ImgCr, OCTET, nTaille);
    lire_image_pgm(cNomImgLue3, ImgCr, nTaille);


    //declaration images réduite

    OCTET *ImgCb_reduite, *ImgCr_reduite;
    allocation_tableau(ImgCb_reduite, OCTET, nTaille/2);
    allocation_tableau(ImgCr_reduite, OCTET, nTaille/2);

    reduce_image_pgm(ImgCb, ImgCb_reduite, nH, nW, 2);
    reduce_image_pgm(ImgCr, ImgCr_reduite, nH, nW, 2);

    // declaration image re sampled

    OCTET *ImgCb_re_sampled, *ImgCr_re_sampled;
    allocation_tableau(ImgCb_re_sampled, OCTET, nTaille);
    allocation_tableau(ImgCr_re_sampled, OCTET, nTaille);

    resample_image_pgm(ImgCb_reduite, ImgCb_re_sampled, nH/2 , nW/2, nH, nW);
    resample_image_pgm(ImgCr_reduite, ImgCr_re_sampled, nH/2 , nW/2, nH, nW);

    // ecriture out

    allocation_tableau(ImgYOut, OCTET, nTaille);
    allocation_tableau(ImgCbOut, OCTET, nTaille);
    allocation_tableau(ImgCrOut, OCTET, nTaille);

    for (int i = 0; i < nTaille; i++) {
        ImgCbOut[i] = ImgCb_re_sampled[i];
        ImgCrOut[i] = ImgCr_re_sampled[i];
    }

    ecrire_image_pgm(cNomImgLue2, ImgCbOut, nH, nW);
    ecrire_image_pgm(cNomImgLue3, ImgCrOut, nH, nW);


}


/* void difference_map_pgm(char cNomImgLue[250], char cNomImgEcrite[250]){
    int nH, nW, nTaille;
    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);

    allocation_tableau(ImgOut, OCTET, nTaille);

    ImgOut[0] = ImgIn[0];

    for (int i = 1; i < nTaille; i++) {
        if ( i % nW == 0 ){
            ImgOut[i] = ImgIn[i] - ImgOut[i - nW] + 128;
        }else {
            ImgOut[i] = ImgIn[i] - ImgOut[i-1] + 128;
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

} */


void entropie(char cNomImgLue[250]){
    int nH, nW, nTaille;
    OCTET *ImgIn;
    
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
    
    printf("Entropie de l'image : %f\n", entropy);
    
    free(ImgIn);
}


void difference_map_pgm(char cNomImgLue[250], char cNomImgEcrite[250]){
    int nH, nW, nTaille;
    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);
    
    allocation_tableau(ImgOut, OCTET, nTaille);

    ImgOut[0] = ImgIn[0];

    for (int i = 1; i < nTaille; i++) {
        if ( i % nW == 0 ){
            ImgOut[i] = ImgIn[i] - ImgIn[i - nW] + 128;
            //std::cout<< "index : " << i << ", valeur : " << (int)ImgOut[i] << std::endl;
        }else {
            ImgOut[i] = ImgIn[i] - ImgIn[i-1] + 128;
            //std::cout<< "index : " << i << ", valeur : " << (int)ImgOut[i] << std::endl;
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

}

void difference_map_pgm_alternate(char cNomImgLue[250], char cNomImgEcrite[250]) {
    int nH, nW, nTaille;
    OCTET *ImgIn, *ImgOut;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);

    allocation_tableau(ImgOut, OCTET, nTaille);

    ImgOut[0] = ImgIn[0];

    for (int i = 1; i < nTaille; i++) {
        int row = i / nW;   
        int col = i % nW;   

        if (row == 0) {
            ImgOut[i] = ImgIn[i] - ImgIn[i - 1] + 128;
        }
        else if (col == 0) {
            ImgOut[i] = ImgIn[i] - ImgIn[i - nW] + 128;
        }
        else {

            int gauche   = ImgIn[i - 1];
            int dessus   = ImgIn[i - nW];
            int diagonal = ImgIn[i - nW - 1];

            int prediction = gauche + dessus - diagonal;

            ImgOut[i] = ImgIn[i] - prediction + 128;
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

    free(ImgIn);
    free(ImgOut);
}

//-------------------------------TP 4-----------------------------------------


void transformee_ondelettes_pgm(char cNomImgLue[250], char cNomImgEcrite[250]) {
    int nH, nW, nTaille;
    OCTET *ImgIn, *ImgOut, *ImgBF, *ImgMFh , *ImgMFv , *ImgHF ;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);
    
    allocation_tableau(ImgOut, OCTET, nTaille);

    allocation_tableau(ImgBF, OCTET, nTaille/4);
    allocation_tableau(ImgMFh, OCTET, nTaille/4);
    allocation_tableau(ImgMFv, OCTET, nTaille/4);
    allocation_tableau(ImgHF, OCTET, nTaille/4);

    for (int i = 0; i < nH / 2; i++) {
        for (int j = 0; j < nW / 2; j++) {
            int A = ImgIn[(i * 2) * nW + (j * 2)];
            int C = ImgIn[((i * 2) + 1) * nW + (j * 2)];
            int B = ImgIn[(i * 2) * nW + ((j * 2) + 1)];
            int D = ImgIn[((i * 2) + 1) * nW + ((j * 2) + 1)];

            ImgBF[i * (nW / 2) + j] = std::min(std::max((A + B + C + D) / 4, 0), 255);
            ImgMFh[i * (nW / 2) + j] = std::min(std::max((A + B - C - D) / 2 + 128, 0), 255);
            ImgMFv[i * (nW / 2) + j] = std::min(std::max((A - B + C - D) / 2 + 128, 0), 255);
            ImgHF[i * (nW / 2) + j] = std::min(std::max((A - B - C + D) + 128, 0), 255);

            ImgOut[(i * 2) * nW + (j * 2)] = ImgBF[i * (nW / 2) + j];
            ImgOut[(i * 2) * nW + ((j * 2) + 1)] = ImgMFh[i * (nW / 2) + j];
            ImgOut[((i * 2) + 1) * nW + (j * 2)] = ImgMFv[i * (nW / 2) + j];
            ImgOut[((i * 2) + 1) * nW + ((j * 2) + 1)] = ImgHF[i * (nW / 2) + j];

        }
    }

    char cNomImgBF[] = "imgBF.pgm";
    char cNomImgMFh[] = "ImgMFh.pgm";
    char cNomImgMFv[] = "ImgMFv.pgm";
    char cNomImgHF[] = "ImgHF.pgm";

    ecrire_image_pgm(cNomImgBF, ImgBF, nH/2, nW/2);
    ecrire_image_pgm(cNomImgMFh, ImgMFh, nH/2, nW/2);
    ecrire_image_pgm(cNomImgMFv, ImgMFv, nH/2, nW/2);
    ecrire_image_pgm(cNomImgHF, ImgHF, nH/2, nW/2);

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

    free(ImgIn);
    free(ImgBF);
    free(ImgMFh);
    free(ImgMFv);
    free(ImgHF);
    free(ImgOut);
}

void reconstruire_ondelettes_pgm(char cNomImgLue[250], char cNomImgEcrite[250]) {
    int nH, nW, nTaille;
    // Lire la taille de l'image (la version transformée a la même taille que l'image d'origine)
    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    // Dimensions des sous-images (ondelettes)
    int nH_sub = nH / 2, nW_sub = nW / 2;
    int subTaille = nH_sub * nW_sub;

    // Allocation et lecture de l'image contenant les sous-bandes
    OCTET *ImgIn, *ImgOut;
    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);

    // Allocation des sous-bandes
    OCTET *ImgBF, *ImgMFh, *ImgMFv, *ImgHF;
    allocation_tableau(ImgBF, OCTET, subTaille);
    allocation_tableau(ImgMFh, OCTET, subTaille);
    allocation_tableau(ImgMFv, OCTET, subTaille);
    allocation_tableau(ImgHF, OCTET, subTaille);

    // Extraction des sous-bandes de l'image lue
    for (int i = 0; i < nH_sub; i++) {
        for (int j = 0; j < nW_sub; j++) {
            ImgBF[i * nW_sub + j] = ImgIn[(i * 2) * nW + (j * 2)];
            ImgMFh[i * nW_sub + j] = ImgIn[(i * 2) * nW + ((j * 2) + 1)];
            ImgMFv[i * nW_sub + j] = ImgIn[((i * 2) + 1) * nW + (j * 2)];
            ImgHF[i * nW_sub + j] = ImgIn[((i * 2) + 1) * nW + ((j * 2) + 1)];
        }
    }

    // Allocation de l'image reconstruite
    allocation_tableau(ImgOut, OCTET, nTaille);

    // Pour chaque bloc 2x2 dans l'image d'origine
    for (int i = 0; i < nH_sub; i++) {
        for (int j = 0; j < nW_sub; j++) {
            int idx = i * nW_sub + j;
            // Récupération des coefficients
            float BF = ImgBF[idx];
            float MFh = ImgMFh[idx];
            float MFv = ImgMFv[idx];
            float HF  = ImgHF[idx];
            
            // Calcul des pixels A, B, C, D du bloc 2x2
            float A = BF + 0.5f * (MFh - 128) + 0.5f * (MFv - 128) + 0.25f * (HF - 128);
            float B = BF + 0.5f * (MFh - 128) - 0.5f * (MFv - 128) - 0.25f * (HF - 128);
            float C = BF - 0.5f * (MFh - 128) + 0.5f * (MFv - 128) - 0.25f * (HF - 128);
            float D = BF - 0.5f * (MFh - 128) - 0.5f * (MFv - 128) + 0.25f * (HF - 128);
            
            // Clamper les valeurs dans [0,255]
            int pixA = std::min(std::max((int)round(A), 0), 255);
            int pixB = std::min(std::max((int)round(B), 0), 255);
            int pixC = std::min(std::max((int)round(C), 0), 255);
            int pixD = std::min(std::max((int)round(D), 0), 255);
            
            // Placer les pixels dans l'image de sortie
            ImgOut[(2 * i) * nW + (2 * j)]         = pixA;
            ImgOut[(2 * i) * nW + (2 * j + 1)]     = pixB;
            ImgOut[(2 * i + 1) * nW + (2 * j)]     = pixC;
            ImgOut[(2 * i + 1) * nW + (2 * j + 1)] = pixD;
        }
    }
    
    // Ecriture de l'image reconstruite
    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);
    
    // Libération de la mémoire
    free(ImgIn);
    free(ImgBF);
    free(ImgMFh);
    free(ImgMFv);
    free(ImgHF);
    free(ImgOut);
}


void transformee_ondelettes_recursive_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int niveauMax, int niveau=0) {
    int nH, nW, nTaille;
    OCTET *ImgIn, *ImgOut, *ImgBF, *ImgMFh , *ImgMFv , *ImgHF ;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);
    
    allocation_tableau(ImgOut, OCTET, nTaille);

    allocation_tableau(ImgBF, OCTET, nTaille/4);
    allocation_tableau(ImgMFh, OCTET, nTaille/4);
    allocation_tableau(ImgMFv, OCTET, nTaille/4);
    allocation_tableau(ImgHF, OCTET, nTaille/4);

    for (int i = 0; i < nH / 2; i++) {
        for (int j = 0; j < nW / 2; j++) {
            int A = ImgIn[(i * 2) * nW + (j * 2)];
            int C = ImgIn[((i * 2) + 1) * nW + (j * 2)];
            int B = ImgIn[(i * 2) * nW + ((j * 2) + 1)];
            int D = ImgIn[((i * 2) + 1) * nW + ((j * 2) + 1)];

            ImgBF[i * (nW / 2) + j] = std::min(std::max((A + B + C + D) / 4, 0), 255);
            ImgMFh[i * (nW / 2) + j] = std::min(std::max((A + B - C - D) / 2 + 128, 0), 255);
            ImgMFv[i * (nW / 2) + j] = std::min(std::max((A - B + C - D) / 2 + 128, 0), 255);
            ImgHF[i * (nW / 2) + j] = std::min(std::max((A - B - C + D) + 128, 0), 255);

            ImgOut[(i * 2) * nW + (j * 2)] = ImgBF[i * (nW / 2) + j];
            ImgOut[(i * 2) * nW + ((j * 2) + 1)] = ImgMFh[i * (nW / 2) + j];
            ImgOut[((i * 2) + 1) * nW + (j * 2)] = ImgMFv[i * (nW / 2) + j];
            ImgOut[((i * 2) + 1) * nW + ((j * 2) + 1)] = ImgHF[i * (nW / 2) + j];
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

    free(ImgIn);
    free(ImgMFh);
    free(ImgMFv);
    free(ImgHF);

    if (niveau < niveauMax) {
        std::string cNomImgBF_str = std::to_string(niveauMax - niveau) + ".pgm";
        char cNomImgBF[250];
        strcpy(cNomImgBF, cNomImgBF_str.c_str());
        ecrire_image_pgm(cNomImgBF, ImgBF, nH/2, nW/2);
        transformee_ondelettes_recursive_pgm(cNomImgBF, cNomImgBF, niveauMax, niveau+1);
    }

    free(ImgBF);
    free(ImgOut);
}

void reconstruire_ondelettes_recursive_pgm(char cNomImgLue[250], char cNomImgEcrite[250], int niveauMax, int niveau=0) {
    int nH, nW, nTaille;
    
    // Si on n'est pas au niveau le plus profond, on commence par reconstruire ce niveau
    if (niveau < niveauMax) {
        std::string cNomImgBF_str = std::to_string(niveauMax - niveau) + ".pgm";
        char cNomImgBF[250];
        strcpy(cNomImgBF, cNomImgBF_str.c_str());
        reconstruire_ondelettes_recursive_pgm(cNomImgBF, cNomImgBF, niveauMax, niveau + 1);
    }

    // Lecture de l'image du niveau courant
    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    int nH_sub = nH / 2, nW_sub = nW / 2;
    int subTaille = nH_sub * nW_sub;

    // Allocation des tableaux
    OCTET *ImgIn, *ImgOut;
    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgIn, nTaille);

    OCTET *ImgBF, *ImgMFh, *ImgMFv, *ImgHF;
    allocation_tableau(ImgBF, OCTET, subTaille);
    allocation_tableau(ImgMFh, OCTET, subTaille);
    allocation_tableau(ImgMFv, OCTET, subTaille);
    allocation_tableau(ImgHF, OCTET, subTaille);

    // Extraction des sous-bandes
    for (int i = 0; i < nH_sub; i++) {
        for (int j = 0; j < nW_sub; j++) {
            ImgBF[i * nW_sub + j] = ImgIn[(i * 2) * nW + (j * 2)];
            ImgMFh[i * nW_sub + j] = ImgIn[(i * 2) * nW + ((j * 2) + 1)];
            ImgMFv[i * nW_sub + j] = ImgIn[((i * 2) + 1) * nW + (j * 2)];
            ImgHF[i * nW_sub + j] = ImgIn[((i * 2) + 1) * nW + ((j * 2) + 1)];
        }
    }

    // Reconstruction
    allocation_tableau(ImgOut, OCTET, nTaille);
    for (int i = 0; i < nH_sub; i++) {
        for (int j = 0; j < nW_sub; j++) {
            float BF = ImgBF[i * nW_sub + j];
            float MFh = ImgMFh[i * nW_sub + j];
            float MFv = ImgMFv[i * nW_sub + j];
            float HF = ImgHF[i * nW_sub + j];

            float A = BF + 0.5f * (MFh - 128) + 0.5f * (MFv - 128) + 0.25f * (HF - 128);
            float B = BF + 0.5f * (MFh - 128) - 0.5f * (MFv - 128) - 0.25f * (HF - 128);
            float C = BF - 0.5f * (MFh - 128) + 0.5f * (MFv - 128) - 0.25f * (HF - 128);
            float D = BF - 0.5f * (MFh - 128) - 0.5f * (MFv - 128) + 0.25f * (HF - 128);

            ImgOut[(2 * i) * nW + (2 * j)] = std::min(std::max((int)round(A), 0), 255);
            ImgOut[(2 * i) * nW + (2 * j + 1)] = std::min(std::max((int)round(B), 0), 255);
            ImgOut[(2 * i + 1) * nW + (2 * j)] = std::min(std::max((int)round(C), 0), 255);
            ImgOut[(2 * i + 1) * nW + (2 * j + 1)] = std::min(std::max((int)round(D), 0), 255);
        }
    }

    // Écriture du résultat
    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

    // Libération de la mémoire
    free(ImgIn);
    free(ImgOut);
    free(ImgBF);
    free(ImgMFh);
    free(ImgMFv);
    free(ImgHF);
}


void analyse_ondelettes_multi_niveaux(char cNomImgOriginale[250]) {
    // Pour chaque niveau de 1 à 6
    for(int N = 1; N <= 6; N++) {
        std::cout << "=== Niveau " << N << " ===" << std::endl;
        
        // Noms des fichiers temporaires
        char cNomImgTransformee[250];
        char cNomImgReconstruite[250];
        sprintf(cNomImgTransformee, "transformee_N%d.pgm", N);
        sprintf(cNomImgReconstruite, "reconstruite_N%d.pgm", N);

        // 1. Transformation
        transformee_ondelettes_recursive_pgm(cNomImgOriginale, cNomImgTransformee, N);

        // 2. Reconstruction
        reconstruire_ondelettes_recursive_pgm(cNomImgTransformee, cNomImgReconstruite, N);

        // 3. Calcul du PSNR
        PSNR_PGM(cNomImgOriginale, cNomImgReconstruite);
    }
}

void quantifier_sousbandes(OCTET* ImgBF, OCTET* ImgMFh, OCTET* ImgMFv, OCTET* ImgHF, 
    int QBF, int QMFh, int QMFv, int QHF, int taille) {
    for(int i = 0; i < taille; i++) {
        ImgBF[i] = (ImgBF[i] / QBF) * QBF;
        ImgMFh[i] = ((ImgMFh[i] - 128) / QMFh) * QMFh + 128;
        ImgMFv[i] = ((ImgMFv[i] - 128) / QMFv) * QMFv + 128;
        ImgHF[i] = ((ImgHF[i] - 128) / QHF) * QHF + 128;
    }
}

void transformee_ondelettes_quantifiee_pgm(char cNomImgLue[250], char cNomImgEcrite[250], 
    int QBF, int QMFh, int QMFv, int QHF) {
int nH, nW, nTaille;
OCTET *ImgIn, *ImgOut, *ImgBF, *ImgMFh, *ImgMFv, *ImgHF;

lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
nTaille = nH * nW;

// Allocation des tableaux
allocation_tableau(ImgIn, OCTET, nTaille);
allocation_tableau(ImgOut, OCTET, nTaille);
allocation_tableau(ImgBF, OCTET, nTaille/4);
allocation_tableau(ImgMFh, OCTET, nTaille/4);
allocation_tableau(ImgMFv, OCTET, nTaille/4);
allocation_tableau(ImgHF, OCTET, nTaille/4);

// Lecture de l'image
lire_image_pgm(cNomImgLue, ImgIn, nTaille);

// Calcul des sous-bandes
for (int i = 0; i < nH / 2; i++) {
for (int j = 0; j < nW / 2; j++) {
int A = ImgIn[(i * 2) * nW + (j * 2)];
int B = ImgIn[(i * 2) * nW + ((j * 2) + 1)];
int C = ImgIn[((i * 2) + 1) * nW + (j * 2)];
int D = ImgIn[((i * 2) + 1) * nW + ((j * 2) + 1)];

ImgBF[i * (nW / 2) + j] = std::min(std::max((A + B + C + D) / 4, 0), 255);
ImgMFh[i * (nW / 2) + j] = std::min(std::max((A + B - C - D) / 2 + 128, 0), 255);
ImgMFv[i * (nW / 2) + j] = std::min(std::max((A - B + C - D) / 2 + 128, 0), 255);
ImgHF[i * (nW / 2) + j] = std::min(std::max((A - B - C + D) + 128, 0), 255);
}
}

// Application de la quantification
quantifier_sousbandes(ImgBF, ImgMFh, ImgMFv, ImgHF, QBF, QMFh, QMFv, QHF, nTaille/4);

// Reconstruction après quantification
for (int i = 0; i < nH / 2; i++) {
for (int j = 0; j < nW / 2; j++) {
float BF = ImgBF[i * (nW / 2) + j];
float MFh = ImgMFh[i * (nW / 2) + j];
float MFv = ImgMFv[i * (nW / 2) + j];
float HF = ImgHF[i * (nW / 2) + j];

float A = BF + 0.5f * (MFh - 128) + 0.5f * (MFv - 128) + 0.25f * (HF - 128);
float B = BF + 0.5f * (MFh - 128) - 0.5f * (MFv - 128) - 0.25f * (HF - 128);
float C = BF - 0.5f * (MFh - 128) + 0.5f * (MFv - 128) - 0.25f * (HF - 128);
float D = BF - 0.5f * (MFh - 128) - 0.5f * (MFv - 128) + 0.25f * (HF - 128);

ImgOut[(2 * i) * nW + (2 * j)] = std::min(std::max((int)round(A), 0), 255);
ImgOut[(2 * i) * nW + (2 * j + 1)] = std::min(std::max((int)round(B), 0), 255);
ImgOut[(2 * i + 1) * nW + (2 * j)] = std::min(std::max((int)round(C), 0), 255);
ImgOut[(2 * i + 1) * nW + (2 * j + 1)] = std::min(std::max((int)round(D), 0), 255);
}
}

// Écriture de l'image reconstruite
ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

// Calcul et affichage du PSNR
std::cout << "PSNR avec quantification (QBF=" << QBF << ", QMFh=" << QMFh 
<< ", QMFv=" << QMFv << ", QHF=" << QHF << "):" << std::endl;
PSNR_PGM(cNomImgLue, cNomImgEcrite);

// Libération de la mémoire
free(ImgIn);
free(ImgOut);
free(ImgBF);
free(ImgMFh);
free(ImgMFv);
free(ImgHF);
}




void transformee_ondelettes_recursive_quantifiee_pgm(char cNomImgLue[250], 
    char cNomImgEcrite[250], 
    int niveauMax, 
    int QBF, int QMFh, int QMFv, int QHF, 
    int niveau=0) {
    
    int nH, nW, nTaille;
    OCTET *ImgIn, *ImgOut, *ImgBF, *ImgMFh, *ImgMFv, *ImgHF;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    // Vérification de la taille minimale
    if (nH < 2 || nW < 2) {
        return;
    }

    allocation_tableau(ImgIn, OCTET, nTaille);
    allocation_tableau(ImgOut, OCTET, nTaille);
    allocation_tableau(ImgBF, OCTET, nTaille/4);
    allocation_tableau(ImgMFh, OCTET, nTaille/4);
    allocation_tableau(ImgMFv, OCTET, nTaille/4);
    allocation_tableau(ImgHF, OCTET, nTaille/4);

    lire_image_pgm(cNomImgLue, ImgIn, nTaille);

    // Calcul des sous-bandes
    for (int i = 0; i < nH / 2; i++) {
        for (int j = 0; j < nW / 2; j++) {
            int A = ImgIn[(i * 2) * nW + (j * 2)];
            int B = ImgIn[(i * 2) * nW + ((j * 2) + 1)];
            int C = ImgIn[((i * 2) + 1) * nW + (j * 2)];
            int D = ImgIn[((i * 2) + 1) * nW + ((j * 2) + 1)];

            ImgBF[i * (nW / 2) + j] = std::min(std::max((A + B + C + D) / 4, 0), 255);
            ImgMFh[i * (nW / 2) + j] = std::min(std::max((A + B - C - D) / 2 + 128, 0), 255);
            ImgMFv[i * (nW / 2) + j] = std::min(std::max((A - B + C - D) / 2 + 128, 0), 255);
            ImgHF[i * (nW / 2) + j] = std::min(std::max((A - B - C + D) + 128, 0), 255);
        }
    }

    // Application de la quantification
    quantifier_sousbandes(ImgBF, ImgMFh, ImgMFv, ImgHF, QBF, QMFh, QMFv, QHF, nTaille/4);

    // Reconstruction de l'image
    for (int i = 0; i < nH / 2; i++) {
        for (int j = 0; j < nW / 2; j++) {
            ImgOut[(i * 2) * nW + (j * 2)] = ImgBF[i * (nW / 2) + j];
            ImgOut[(i * 2) * nW + ((j * 2) + 1)] = ImgMFh[i * (nW / 2) + j];
            ImgOut[((i * 2) + 1) * nW + (j * 2)] = ImgMFv[i * (nW / 2) + j];
            ImgOut[((i * 2) + 1) * nW + ((j * 2) + 1)] = ImgHF[i * (nW / 2) + j];
        }
    }

    // Appel récursif sur la sous-bande BF si nécessaire
    if (niveau < niveauMax - 1 && nH >= 4 && nW >= 4) {
        char temp_BF[250];
        sprintf(temp_BF, "temp_BF_N%d_%d.pgm", niveauMax, niveau + 1);
        
        ecrire_image_pgm(temp_BF, ImgBF, nH/2, nW/2);
        transformee_ondelettes_recursive_quantifiee_pgm(temp_BF, temp_BF, 
            niveauMax, QBF, QMFh, QMFv, QHF, niveau + 1);
            
        lire_image_pgm(temp_BF, ImgBF, nTaille/4);
        
        // Mise à jour de l'image de sortie avec la nouvelle sous-bande BF
        for (int i = 0; i < nH/2; i++) {
            for (int j = 0; j < nW/2; j++) {
                ImgOut[(i * 2) * nW + (j * 2)] = ImgBF[i * (nW/2) + j];
            }
        }
        
        remove(temp_BF);
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

    // Libération de la mémoire
    free(ImgIn);
    free(ImgOut);
    free(ImgBF);
    free(ImgMFh);
    free(ImgMFv);
    free(ImgHF);
}


void reconstruire_ondelettes_recursive_quantifiee_pgm(char cNomImgLue[250], 
    char cNomImgEcrite[250], 
    int niveauMax,
    int QBF, int QMFh, int QMFv, int QHF,
    int niveau=0) {
    
    int nH, nW, nTaille;
    
    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    int nH_sub = nH / 2, nW_sub = nW / 2;
    int subTaille = nH_sub * nW_sub;

    OCTET *ImgIn, *ImgOut;
    OCTET *ImgBF, *ImgMFh, *ImgMFv, *ImgHF;
    
    allocation_tableau(ImgIn, OCTET, nTaille);
    allocation_tableau(ImgOut, OCTET, nTaille);
    allocation_tableau(ImgBF, OCTET, subTaille);
    allocation_tableau(ImgMFh, OCTET, subTaille);
    allocation_tableau(ImgMFv, OCTET, subTaille);
    allocation_tableau(ImgHF, OCTET, subTaille);

    lire_image_pgm(cNomImgLue, ImgIn, nTaille);

    // Extraction des sous-bandes
    for (int i = 0; i < nH_sub; i++) {
        for (int j = 0; j < nW_sub; j++) {
            ImgBF[i * nW_sub + j] = ImgIn[(i * 2) * nW + (j * 2)];
            ImgMFh[i * nW_sub + j] = ImgIn[(i * 2) * nW + ((j * 2) + 1)];
            ImgMFv[i * nW_sub + j] = ImgIn[((i * 2) + 1) * nW + (j * 2)];
            ImgHF[i * nW_sub + j] = ImgIn[((i * 2) + 1) * nW + ((j * 2) + 1)];
        }
    }

    // Traitement récursif sur la sous-bande BF si nécessaire
    if (niveau < niveauMax - 1 && nH_sub >= 2 && nW_sub >= 2) {
        char temp_BF[250];
        sprintf(temp_BF, "temp_BF_N%d_%d.pgm", niveauMax, niveau + 1);
        
        ecrire_image_pgm(temp_BF, ImgBF, nH_sub, nW_sub);
        reconstruire_ondelettes_recursive_quantifiee_pgm(temp_BF, temp_BF,
            niveauMax, QBF, QMFh, QMFv, QHF, niveau + 1);
            
        lire_image_pgm(temp_BF, ImgBF, subTaille);
        
        remove(temp_BF);
    }


    // Application de la quantification inverse
    quantifier_sousbandes(ImgBF, ImgMFh, ImgMFv, ImgHF, QBF, QMFh, QMFv, QHF, subTaille);

    // Reconstruction
    for (int i = 0; i < nH_sub; i++) {
        for (int j = 0; j < nW_sub; j++) {
            float BF = ImgBF[i * nW_sub + j];
            float MFh = ImgMFh[i * nW_sub + j] - 128.0f;
            float MFv = ImgMFv[i * nW_sub + j] - 128.0f;
            float HF = ImgHF[i * nW_sub + j] - 128.0f;

            float A = BF + 0.5f * MFh + 0.5f * MFv + 0.25f * HF;
            float B = BF + 0.5f * MFh - 0.5f * MFv - 0.25f * HF;
            float C = BF - 0.5f * MFh + 0.5f * MFv - 0.25f * HF;
            float D = BF - 0.5f * MFh - 0.5f * MFv + 0.25f * HF;

            ImgOut[(2 * i) * nW + (2 * j)] = std::min(std::max((int)round(A), 0), 255);
            ImgOut[(2 * i) * nW + (2 * j + 1)] = std::min(std::max((int)round(B), 0), 255);
            ImgOut[(2 * i + 1) * nW + (2 * j)] = std::min(std::max((int)round(C), 0), 255);
            ImgOut[(2 * i + 1) * nW + (2 * j + 1)] = std::min(std::max((int)round(D), 0), 255);
        }
    }

    ecrire_image_pgm(cNomImgEcrite, ImgOut, nH, nW);

    // Libération de la mémoire
    free(ImgIn); free(ImgOut);
    free(ImgBF); free(ImgMFh); free(ImgMFv); free(ImgHF);
}

void analyse_ondelettes_quantifiees_multi_niveaux(char cNomImgOriginale[250], 
    int QBF, int QMFh, int QMFv, int QHF) {
    // Pour chaque niveau de 1 à 6
    for(int N = 1; N <= 6; N++) {
        std::cout << "=== Niveau " << N << " avec quantification ===" << std::endl;
        std::cout << "QBF=" << QBF << ", QMFh=" << QMFh 
                 << ", QMFv=" << QMFv << ", QHF=" << QHF << std::endl;

        // Noms des fichiers temporaires
        char cNomImgTransformee[250];
        char cNomImgReconstruite[250];
        sprintf(cNomImgTransformee, "transformee_quantif_N%d.pgm", N);
        sprintf(cNomImgReconstruite, "reconstruite_quantif_N%d.pgm", N);

        // Ajuster les facteurs de quantification en fonction du niveau
        int QBF_N = QBF * (N);  // Quantification plus forte pour les niveaux profonds
        int QMFh_N = QMFh * (N);
        int QMFv_N = QMFv * (N);
        int QHF_N = QHF * (N);

        // 1. Transformation avec quantification récursive
        transformee_ondelettes_recursive_quantifiee_pgm(cNomImgOriginale, 
                cNomImgTransformee, 
                N, QBF_N, QMFh_N, QMFv_N, QHF_N);

        // 2. Reconstruction
        reconstruire_ondelettes_recursive_quantifiee_pgm(cNomImgTransformee, 
            cNomImgReconstruite, 
            N, QBF_N, QMFh_N, QMFv_N, QHF_N);

        // 3. Calcul du PSNR
        std::cout << "PSNR pour niveau " << N << ": ";
        PSNR_PGM(cNomImgOriginale, cNomImgReconstruite);
        std::cout << std::endl;
    }
}

void analyse_debit_distorsion(char cNomImgOriginale[250]) {
    std::cout << "Format: Niveau,QBF,QMFh,QMFv,QHF,BPP,PSNR" << std::endl;
    
    // Plus de valeurs de quantification pour plus de points
    int Q_values[] = {2, 4, 8, 12, 16, 24, 32, 48, 64};
    int num_q_values = 9;
    
    for(int N = 1; N <= 6; N++) {
        for(int q_idx = 0; q_idx < num_q_values; q_idx++) {
            int QBF = Q_values[q_idx];
            int QMFh = Q_values[q_idx] * 2;
            int QMFv = Q_values[q_idx] * 2;
            int QHF = Q_values[q_idx] * 4;

            char cNomImgTransformee[250];
            char cNomImgReconstruite[250];
            sprintf(cNomImgTransformee, "transformee_N%d_Q%d.pgm", N, QBF);
            sprintf(cNomImgReconstruite, "reconstruite_N%d_Q%d.pgm", N, QBF);

            // Transformation et reconstruction
            transformee_ondelettes_recursive_quantifiee_pgm(
                cNomImgOriginale, cNomImgTransformee, 
                N, QBF, QMFh, QMFv, QHF);
            
            reconstruire_ondelettes_recursive_quantifiee_pgm(
                cNomImgTransformee, cNomImgReconstruite, 
                N, QBF, QMFh, QMFv, QHF);

            // Calcul du débit plus précis
            int nH, nW, nTaille;
            lire_nb_lignes_colonnes_image_pgm(cNomImgOriginale, &nH, &nW);
            nTaille = nH * nW;
            
            // Calcul du débit théorique basé sur la quantification
            float bits_per_coeff = 0;
            if (N == 1) {
                bits_per_coeff = (8.0f/4) * (1.0f/QBF + 1.0f/QMFh + 1.0f/QMFv + 1.0f/QHF);
            } else {
                // Pour les niveaux supérieurs, prendre en compte la récursion
                float factor = 1.0f / (4 * N);
                bits_per_coeff = 8.0f * (factor/QBF + factor/QMFh + factor/QMFv + factor/QHF);
            }
            
            float bpp = bits_per_coeff * nTaille / nTaille;  // bits par pixel

            // Afficher les résultats
            std::cout << N << "," << QBF << "," << QMFh << "," 
                     << QMFv << "," << QHF << "," << bpp << ",";
            PSNR_PGM(cNomImgOriginale, cNomImgReconstruite);
        }
    }
}

void RGB_to_YCrCb(OCTET* ImgRGB, OCTET* ImgY, OCTET* ImgCr, OCTET* ImgCb, int nTaille) {
    for(int i = 0; i < nTaille/3; i++) {
        float r = ImgRGB[3*i];
        float g = ImgRGB[3*i+1];
        float b = ImgRGB[3*i+2];

        // Formules standards de conversion RGB -> YCrCb
        float Y = 0.299f * r + 0.587f * g + 0.114f * b;
        float Cb = -0.1687f * r - 0.3313f * g + 0.5f * b + 128;
        float Cr = 0.5f * r - 0.4187f * g - 0.0813f * b + 128;

        ImgY[i] = (OCTET)std::min(std::max((int)(Y), 0), 255);
        ImgCb[i] = (OCTET)std::min(std::max((int)(Cb), 0), 255);
        ImgCr[i] = (OCTET)std::min(std::max((int)(Cr), 0), 255);
    }
}

void YCrCb_to_RGB(OCTET* ImgY, OCTET* ImgCr, OCTET* ImgCb, OCTET* ImgRGB, int nTaille) {
    for (int i = 0; i < nTaille/3; i++) {
        float Y = ImgY[i];
        float Cb = ImgCb[i] - 128.0f;
        float Cr = ImgCr[i] - 128.0f;

        // Formules standards de conversion YCrCb -> RGB
        float R = Y + 1.402f * Cr;
        float G = Y - 0.34414f * Cb - 0.71414f * Cr;
        float B = Y + 1.772f * Cb;

        ImgRGB[3*i] = (OCTET)std::min(std::max((int)(R), 0), 255);
        ImgRGB[3*i+1] = (OCTET)std::min(std::max((int)(G), 0), 255);
        ImgRGB[3*i+2] = (OCTET)std::min(std::max((int)(B), 0), 255);
    }
}

void transformee_ondelettes_recursive_quantifiee_ppm(char cNomImgLue[250], 
    char cNomImgEcrite[250], 
    int niveauMax, 
    int QBF_Y, int QMFh_Y, int QMFv_Y, int QHF_Y,
    int QBF_C, int QMFh_C, int QMFv_C, int QHF_C,
    int niveau=0) {
    
    int nH, nW, nTaille, nTaille3;
    
    lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;
    nTaille3 = nTaille * 3;

    // Allocation et lecture
    OCTET *ImgRGB, *ImgY, *ImgCr, *ImgCb;
    allocation_tableau(ImgRGB, OCTET, nTaille3);
    allocation_tableau(ImgY, OCTET, nTaille);
    allocation_tableau(ImgCr, OCTET, nTaille);
    allocation_tableau(ImgCb, OCTET, nTaille);

    lire_image_ppm(cNomImgLue, ImgRGB, nTaille);
    RGB_to_YCrCb(ImgRGB, ImgY, ImgCr, ImgCb, nTaille3);

    // Noms des fichiers temporaires
    char temp_Y[250], temp_Cr[250], temp_Cb[250];
    char temp_Y_trans[250], temp_Cr_trans[250], temp_Cb_trans[250];
    char temp_Y_rec[250], temp_Cr_rec[250], temp_Cb_rec[250];
    
    sprintf(temp_Y, "temp_Y_N%d_%d.pgm", niveauMax, niveau);
    sprintf(temp_Cr, "temp_Cr_N%d_%d.pgm", niveauMax, niveau);
    sprintf(temp_Cb, "temp_Cb_N%d_%d.pgm", niveauMax, niveau);
    
    sprintf(temp_Y_trans, "temp_Y_trans_N%d_%d.pgm", niveauMax, niveau);
    sprintf(temp_Cr_trans, "temp_Cr_trans_N%d_%d.pgm", niveauMax, niveau);
    sprintf(temp_Cb_trans, "temp_Cb_trans_N%d_%d.pgm", niveauMax, niveau);
    
    sprintf(temp_Y_rec, "temp_Y_rec_N%d_%d.pgm", niveauMax, niveau);
    sprintf(temp_Cr_rec, "temp_Cr_rec_N%d_%d.pgm", niveauMax, niveau);
    sprintf(temp_Cb_rec, "temp_Cb_rec_N%d_%d.pgm", niveauMax, niveau);

    // ... reste du

    // Sauvegarder les composantes
    ecrire_image_pgm(temp_Y, ImgY, nH, nW);
    ecrire_image_pgm(temp_Cr, ImgCr, nH, nW);
    ecrire_image_pgm(temp_Cb, ImgCb, nH, nW);

    // Transformation et reconstruction pour chaque composante
    transformee_ondelettes_recursive_quantifiee_pgm(temp_Y, temp_Y_trans, niveauMax, 
        QBF_Y, QMFh_Y, QMFv_Y, QHF_Y, niveau);
    reconstruire_ondelettes_recursive_quantifiee_pgm(temp_Y_trans, temp_Y, niveauMax,
        QBF_Y, QMFh_Y, QMFv_Y, QHF_Y, niveau);

    transformee_ondelettes_recursive_quantifiee_pgm(temp_Cr, temp_Cr_trans, niveauMax,
        QBF_C, QMFh_C, QMFv_C, QHF_C, niveau);
    reconstruire_ondelettes_recursive_quantifiee_pgm(temp_Cr_trans, temp_Cr, niveauMax,
        QBF_C, QMFh_C, QMFv_C, QHF_C, niveau);

    transformee_ondelettes_recursive_quantifiee_pgm(temp_Cb, temp_Cb_trans, niveauMax,
        QBF_C, QMFh_C, QMFv_C, QHF_C, niveau);
    reconstruire_ondelettes_recursive_quantifiee_pgm(temp_Cb_trans, temp_Cb, niveauMax,
        QBF_C, QMFh_C, QMFv_C, QHF_C, niveau);

    // Lecture des résultats reconstruits
    lire_image_pgm(temp_Y, ImgY, nTaille);
    lire_image_pgm(temp_Cr, ImgCr, nTaille);
    lire_image_pgm(temp_Cb, ImgCb, nTaille);

    // Reconversion en RGB et sauvegarde
    YCrCb_to_RGB(ImgY, ImgCr, ImgCb, ImgRGB, nTaille3);
    ecrire_image_ppm(cNomImgEcrite, ImgRGB, nH, nW);

    // Nettoyage
    free(ImgRGB); free(ImgY); free(ImgCr); free(ImgCb);
    // Nettoyage plus complet
    remove(temp_Y); remove(temp_Y_trans); remove(temp_Y_rec);
    remove(temp_Cr); remove(temp_Cr_trans); remove(temp_Cr_rec);
    remove(temp_Cb); remove(temp_Cb_trans); remove(temp_Cb_rec);
}

void analyse_debit_distorsion_ppm(char cNomImgOriginale[250]) {
    std::cout << "Format: Niveau,QBF_Y,QMFh_Y,QMFv_Y,QHF_Y,QBF_C,QMFh_C,QMFv_C,QHF_C,BPP,PSNR" << std::endl;
    
    int Q_values[] = {2, 4, 8, 12, 16, 24, 32, 48, 64};
    int num_q_values = 9;
    
    for(int N = 1; N <= 6; N++) {
        for(int q_idx = 0; q_idx < num_q_values; q_idx++) {
            // Facteurs pour la luminance (Y)
            int QBF_Y = Q_values[q_idx];
            int QMFh_Y = Q_values[q_idx] * 2;
            int QMFv_Y = Q_values[q_idx] * 2;
            int QHF_Y = Q_values[q_idx] * 4;

            // Facteurs pour la chrominance (Cr,Cb)
            int QBF_C = Q_values[q_idx] * 2;
            int QMFh_C = Q_values[q_idx] * 4;
            int QMFv_C = Q_values[q_idx] * 4;
            int QHF_C = Q_values[q_idx] * 8;

            char cNomImgReconstruite[250];
            sprintf(cNomImgReconstruite, "reconstruite_N%d_Q%d.ppm", N, QBF_Y);

            // Transformation et reconstruction directe
            transformee_ondelettes_recursive_quantifiee_ppm(
                cNomImgOriginale, cNomImgReconstruite, 
                N, 
                QBF_Y, QMFh_Y, QMFv_Y, QHF_Y,
                QBF_C, QMFh_C, QMFv_C, QHF_C);

            // Calcul du débit théorique (3 composantes)
            float bpp_y = (8.0f/(4*N)) * (1.0f/QBF_Y + 1.0f/QMFh_Y + 1.0f/QMFv_Y + 1.0f/QHF_Y);
            float bpp_c = (8.0f/(4*N)) * (1.0f/QBF_C + 1.0f/QMFh_C + 1.0f/QMFv_C + 1.0f/QHF_C);
            float bpp = bpp_y + 2 * bpp_c;  // Y + Cr + Cb

            // Afficher les résultats
            std::cout << N << "," << QBF_Y << "," << QMFh_Y << "," 
                     << QMFv_Y << "," << QHF_Y << "," << QBF_C << ","
                     << QMFh_C << "," << QMFv_C << "," << QHF_C << ","
                     << bpp << ",";
            PSNR_PPM(cNomImgOriginale, cNomImgReconstruite);
        }
    }
}

#endif // CODAGE_h