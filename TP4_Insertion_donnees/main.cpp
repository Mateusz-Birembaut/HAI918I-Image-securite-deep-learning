#include "../src/image_ppm.h"
#include "../src/codage.h"
#include "Insertion.h"


int main(){

    //extractBinaryPlans("catOFB.pgm");
/* 

    char nameBuffer[300];
    char* names[8];

    for (int i = 0; i < 8; ++i) {
        sprintf(nameBuffer, "img_msg_%d.pgm", i);
        names[i] = strdup(nameBuffer);     
    }

    for(int i {0}; i < 8; ++i){
        addMsgToImage("cat.pgm", names[i], i);
    } 

    for (int i = 0; i < 8; ++i) {
        sprintf(nameBuffer, "img_msg_%d.pgm", i);
        PSNR_PGM("cat.pgm", nameBuffer); 
    }  */

    histogramme_pgm("cat.pgm", "M_Hist.dat");
    histogramme_pgm("img_msg_0.pgm", "M_ECB_Hist.dat");
 
    return EXIT_SUCCESS;
}