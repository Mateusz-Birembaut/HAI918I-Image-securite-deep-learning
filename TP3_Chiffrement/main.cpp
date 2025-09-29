#include "../src/image_ppm.h"
#include "../src/codage.h"
#include "Chiffrement.h"

// g++ main.cpp -D USE_CXX_AES -O3 -std=c++17

int main(){

    /*
    encryptAES("cat.pgm", "catECB.pgm", 16);

    std::cout << "Image ECB : \n";
    PSNR_PGM("cat.pgm", "catECB.pgm");
    entropie("catECB.pgm");

    std::vector<unsigned char> vecteur_init {1,52,42,254,36,45,1,52,42,254,36,45, 17, 85, 99, 125};

    std::cout << "Image CBC : \n";
    encryptCBC("cat.pgm", "catCBC.pgm", vecteur_init);
    PSNR_PGM("cat.pgm", "catCBC.pgm");
    entropie("catCBC.pgm");

    std::cout << "Image CFB : \n";
    encryptCFB("cat.pgm", "catCFB.pgm", vecteur_init);
    PSNR_PGM("cat.pgm", "catCFB.pgm");
    entropie("cat.pgm");

    std::cout << "Image OFB : \n";
    encryptOFB("cat.pgm", "catOFB.pgm", vecteur_init);
    PSNR_PGM("cat.pgm", "catOFB.pgm");
    entropie("catOFB.pgm");

    std::cout << "Image CTR : \n";
    encryptCTR("cat.pgm", "catCTR.pgm", vecteur_init);
    PSNR_PGM("cat.pgm", "catCTR.pgm");
    entropie("catCTR.pgm");
    */
    std::cout << "-------------------------------------------------\n";

    /*
    encryptAES("Medical.pgm", "MedicalECB.pgm", 16);

    std::cout << "Image ECB : \n";
    PSNR_PGM("Medical.pgm", "MedicalECB.pgm");
    entropie("MedicalECB.pgm");

    std::vector<unsigned char> vecteur_init {1,52,42,254,36,45,1,52,42,254,36,45, 17, 85, 99, 125};

    std::cout << "Image CBC : \n";
    encryptCBC("Medical.pgm", "MedicalCBC.pgm", vecteur_init);
    PSNR_PGM("Medical.pgm", "MedicalCBC.pgm");
    entropie("MedicalCBC.pgm");

    std::cout << "Image CFB : \n";
    encryptCFB("Medical.pgm", "MedicalCFB.pgm", vecteur_init);
    PSNR_PGM("Medical.pgm", "MedicalCFB.pgm");
    entropie("MedicalCFB.pgm");

    std::cout << "Image OFB : \n";
    encryptOFB("Medical.pgm", "MedicalOFB.pgm", vecteur_init);
    PSNR_PGM("Medical.pgm", "MedicalOFB.pgm");
    entropie("MedicalOFB.pgm");

    std::cout << "Image CTR : \n";
    encryptCTR("Medical.pgm", "MedicalCTR.pgm", vecteur_init);
    PSNR_PGM("Medical.pgm", "MedicalCTR.pgm");
    entropie("MedicalCTR.pgm");

    decryptAES("MedicalECB.pgm", "MedicalECBDecrypt.pgm", 16);
    PSNR_PGM("Medical.pgm", "MedicalECBDecrypt.pgm");
    entropie("MedicalECBDecrypt.pgm");
    */

    std::cout << "-------------------------------------------------\n";

    /*
    decryptAES("catECB.pgm", "catECBDecrypt.pgm", 16); // ok 
    PSNR_PGM("cat.pgm", "catECBDecrypt.pgm");

    decryptCBC("catCBC.pgm", "catCBCDecrypt.pgm", vecteur_init);
    PSNR_PGM("cat.pgm", "catCBCDecrypt.pgm");

    decryptCFB("catCFB.pgm", "catCFBDecrypt.pgm", vecteur_init); // ok
    PSNR_PGM("cat.pgm", "catCFBDecrypt.pgm");

    decryptOFB("catOFB.pgm", "catOFBDecrypt.pgm", vecteur_init); // ok 
    PSNR_PGM("cat.pgm", "catOFBDecrypt.pgm");

    decryptCTR("catCTR.pgm", "catCTRDecrypt.pgm", vecteur_init);
    PSNR_PGM("cat.pgm", "catCTRDecrypt.pgm");
    */


    histogramme_pgm("Medical.pgm", "M_Hist.dat");
    histogramme_pgm("MedicalECB.pgm", "M_ECB_Hist.dat");

    return EXIT_SUCCESS;
}