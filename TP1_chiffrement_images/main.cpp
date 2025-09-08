#include "chiffrement_images.h"
#include "../src/codage.h"

int main(){
	/* 
	histogramme_pgm("../i.pgm" , "i.dat");

	permutation("../i.pgm", "../iPermut.pgm", 'a');
	
	PSNR_PGM("../i.pgm", "../iPermut.pgm");
	*/

	//histogramme_pgm("../iPermut.pgm" , "perm.dat");

	/*
	entropie("../iPermut.pgm");
 
	permutation_inv("../iPermut.pgm", "../iPermut_inv.pgm", 'a');

	substitution("../i.pgm", "../iSubstitution.pgm", 'z');

	PSNR_PGM("../i.pgm", "../iSubstitution.pgm");
	entropie("../iSubstitution.pgm");

	histogramme_pgm("../iSubstitution.pgm" , "sub.dat");
	*/
	// substitution_inv("../iSubstitution.pgm", "../iSubstitution_inv.pgm", 'z'); 

	substitution("../cat.pgm", "../chiffre.pgm", 218);
	brut_force_substitution("../chiffre.pgm", "../i_bruteForce.pgm");

	return EXIT_SUCCESS;
}

