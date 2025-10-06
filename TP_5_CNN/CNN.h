#include "../src/image_ppm.h"
#include "../src/codage.h"

#include <filesystem>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>

std::mt19937 gen(20);

inline OCTET clamp(int value, int min = 0, int max = 255) {
    return value < min ? min : (value > max ? max : value);
}

struct Kernel{

    Kernel(int _size, std::vector<int>&& _values) 
        : size(_size), values(std::move(_values)) {
        if (values.size() != size * size) {
            throw std::invalid_argument("Le nombre de valeurs ne correspond pas à la taille du noyau.");
        }
    }

    float sumValues(){
        float sum {};
        for(auto val : values){
            sum += val;
        }
        return sum;
    }

    int size; 
    std::vector<int> values;
};

struct Layer{
    std::vector<std::vector<float>> weightsInputs;
    std::vector<float> values;
    std::vector<float> proba;

    Layer(const std::vector<float>& inputs, int nbOfNodes){
        values.resize(nbOfNodes);
        proba.resize(nbOfNodes);
        weightsInputs.resize(inputs.size());

        // Générateur de nombres aléatoires

        std::uniform_real_distribution<float> dis(-0.05f, 0.05f); 

        // each input ajouter poid en mode fully connected
        for(int i {0}; i < inputs.size(); ++i){
            weightsInputs[i].resize(nbOfNodes);
            for(int j{0}; j < nbOfNodes; j++){
                weightsInputs[i][j] = dis(gen);//1.0f/(nbOfNodes * inputs.size());
            }
        }

        computeWeights(inputs);
    }

    void computeWeights(const std::vector<float>& inputs){
        for(int i {0}; i < values.size(); i++){
            float sum {};
            for(int j {0}; j < inputs.size(); j++){
                sum += weightsInputs[j][i] * inputs[j];
            }
            //std::cout << sum << '\n';
            values[i] = sum;
        }
    }

    float sumValues(){
        float sum {};
        for(int i {0}; i < values.size(); i++){
            sum += exp(values[i]);
        }
        return sum;
    }

    void softmax(){
        float sumExpInv {1.0f/sumValues()};

        for(int i {0}; i < proba.size(); i++){
            proba[i] = exp(values[i]) * sumExpInv;
            //std::cout << proba[i]  << '\n';
        }

    }

};

using FilterFunction = OCTET (*)(OCTET*, int, int, int, int, Kernel);

OCTET convolution_function(OCTET* image, int nW, int nH, int ligne, int colonne, Kernel kernel){
    int sum = 0;

    if( ligne < kernel.size/2 || colonne < kernel.size/2 || colonne >= nW - kernel.size/2 || ligne >= nH - kernel.size/2 ) return 0;

	for (int i = -kernel.size/2; i <= kernel.size/2; i++) {
		for (int j = -kernel.size/2; j <= kernel.size/2; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;

			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
                int kernelIndex = (i + kernel.size / 2) * kernel.size + (j + kernel.size / 2);
                sum += image[voisinLigne * nW + voisinColonne] * kernel.values[kernelIndex];
			}
		}
	}
	return clamp(std::round(sum / kernel.sumValues()));
}

OCTET* convolution(OCTET* pt_image, int nH, int nW, Kernel kernel, FilterFunction filter) {
	int nTaille;

	OCTET* ImgOut;

	nTaille = nH * nW;
	allocation_tableau(ImgOut, OCTET, nTaille);

	for (int i = 0; i < nH; i++) {
		for (int j = 0; j < nW; j++) {
			ImgOut[i * nW + j] = filter(pt_image, nW, nH, i, j, kernel);
		}
	}

	return ImgOut;
}

OCTET pool_function(OCTET* image, int nW, int nH, int ligne, int colonne, Kernel kernel){
    int max = 0;
	for (int i = 0; i < kernel.size; i++) {
		for (int j = 0; j < kernel.size; j++) {
			int voisinLigne = ligne + i;
			int voisinColonne = colonne + j;

			if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW) {
                auto value = image[voisinLigne * nW + voisinColonne];
                if (value > max) max = value;
			}
		}
	}
	return max;
}

void pooling(OCTET*& ImgIn, int nH, int nW, Kernel kernel, FilterFunction filter){
    OCTET* ImgOut;

    int nH2 = (nH/kernel.size) -1;
    int nW2 = (nW/kernel.size) -1;

	int nTaille = nH2 * nW2;
	allocation_tableau(ImgOut, OCTET, nTaille);

    for (int i = 1; i < nH2-1; i++) {
		for (int j = 1; j < nW2-1; j++) {
			ImgOut[i * nW2 + j] = filter(ImgIn, nH, nW, i*kernel.size, j*kernel.size, kernel);
		}
	}

    ImgIn = ImgOut;
}


std::vector<float> test(char nom[250]) {
    int nH, nW, nTaille;

    std::vector<Kernel> kernels_pass1 = {
        Kernel(3, std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1, 1}), 
        Kernel(3, std::vector<int>{1, 2, 1, 2, 4, 2, 1, 2, 1}),

        Kernel(3, std::vector<int>{0, -1, 0, -1, 5, -1, 0, -1, 0}), 
        Kernel(3, std::vector<int>{0, -1, 0, -1, 10, -1, 0, -1, 0}), 

        //Kernel(3, std::vector<int>{4, -4, -3, -1, 5, -5, 2, -1, 3})  
    };


    std::vector<OCTET*> images(kernels_pass1.size());

    FilterFunction filter = &convolution_function;

    OCTET *ImgInC1; 

    lire_nb_lignes_colonnes_image_pgm(nom, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgInC1, OCTET, nTaille);

    lire_image_pgm(nom, ImgInC1, nTaille);

    for(int j = 0; j < kernels_pass1.size(); j++) {  

        OCTET *ImgOutC1;

        allocation_tableau(ImgOutC1, OCTET, nTaille);

        ImgOutC1 = convolution(ImgInC1, nH, nW, kernels_pass1[j], filter);  

        images[j] = ImgOutC1;  
    }
    
    FilterFunction pool_fun = &pool_function;
    Kernel poolKernel {2, std::vector<int>{1,1,1,1}};

    for(int i{0}; i < images.size(); ++i){
        pooling(images[i], nH, nW, poolKernel, pool_fun);
    }

    nH /= poolKernel.size;
    nW /= poolKernel.size;
    nH--;
    nW--;

    nTaille = nH * nW;

    // -------------------------PASS 2-----------------
    std::vector<Kernel> kernels_pass2 {
        Kernel(3, std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1, 1}), 

        Kernel(3, std::vector<int>{0, -1, 0, -1, 5, -1, 0, -1, 0}), 

        Kernel(3, std::vector<int>{-1, -5, 4, 2, 3, -4, -6, 1, 5})  
    };

    std::vector<OCTET*> images_pass2(images.size() * kernels_pass2.size());

    int outputIndex = 0;
    for(int i{0}; i < images.size(); ++i){
        for(int j{0} ; j < kernels_pass2.size(); j++){

            OCTET *ImgOut;

            allocation_tableau(ImgOut, OCTET, nTaille);

            ImgOut = convolution(images[i], nH, nW, kernels_pass2[j], filter);  

            images_pass2[outputIndex] = ImgOut;  
            outputIndex++;
        }
        free(images[i]);
    }

    for(int i{0}; i < images_pass2.size(); ++i){
        pooling(images_pass2[i], nH, nW, poolKernel, pool_fun);
    }


    nH /= poolKernel.size;
    nW /= poolKernel.size;
    nH--;
    nW--;

    nTaille = nH * nW;

    std::vector<float> data;
    data.reserve((images_pass2.size() * nTaille));

    // mettre a plat
    for(int i{0}; i < images_pass2.size(); ++i){
        for(int j{0}; j < nTaille ; ++j){
            data.push_back(images_pass2[i][j]);
        }
        free(images_pass2[i]);
    }

    //std::cout << "Nombre d'éléments apres la mise a plat : "<< data.size() << '\n';

    // couche fully connected entre les data et 2 noeuds

    Layer finalLayer(data, 2);
    finalLayer.softmax();

    return finalLayer.proba;
}

void grade(){
    std::string base_path = "images/";
    std::vector<std::string> class_0_images;
    std::vector<std::string> class_1_images;
    
    // Récupérer les images de la classe 0
    std::string class_0_path = base_path + "class_0";
    if (std::filesystem::exists(class_0_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(class_0_path)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension();
                if (extension == ".pgm" || extension == ".ppm") {
                    class_0_images.push_back(entry.path().string());
                }
            }
        }
    }
    
    // Récupérer les images de la classe 1
    std::string class_1_path = base_path + "class_1";
    if (std::filesystem::exists(class_1_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(class_1_path)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension();
                if (extension == ".pgm" || extension == ".ppm") {
                    class_1_images.push_back(entry.path().string());
                }
            }
        }
    }
    
    // Mélanger les images pour avoir un échantillonnage aléatoire
    std::shuffle(class_0_images.begin(), class_0_images.end(), gen);
    std::shuffle(class_1_images.begin(), class_1_images.end(), gen);
    
    // Prendre seulement 80 images de chaque classe
    if (class_0_images.size() > 80) {
        class_0_images.resize(80);
    }
    if (class_1_images.size() > 80) {
        class_1_images.resize(80);
    }
    
    std::cout << "Classe 0: " << class_0_images.size() << " images trouvées" << std::endl;
    std::cout << "Classe 1: " << class_1_images.size() << " images trouvées" << std::endl;
    
    // Variables pour les métriques
    int TP = 0, TN = 0, FP = 0, FN = 0;
    
    // Traiter les images de la classe 0 (vraie classe = 0)
    std::cout << "Traitement des images de la classe 0..." << std::endl;
    for (const auto& image_path : class_0_images) {
        char nom[250];
        strcpy(nom, image_path.c_str());
        std::vector<float> probabilities = test(nom);
        
        // Prédiction : classe avec la plus haute probabilité
        int predicted_class = (probabilities[0] > probabilities[1]) ? 0 : 1;
        int true_class = 0;
        
        // Mise à jour des métriques
        if (true_class == 0 && predicted_class == 0) {
            TN++; // Vrai Négatif
        } else if (true_class == 0 && predicted_class == 1) {
            FP++; // Faux Positif
        }
        
        std::cout << "Image: " << image_path << " - Vraie classe: " << true_class 
                  << ", Prédite: " << predicted_class << " (P0: " << probabilities[0] 
                  << ", P1: " << probabilities[1] << ")" << std::endl;
    }
    
    // Traiter les images de la classe 1 (vraie classe = 1)
    std::cout << "Traitement des images de la classe 1..." << std::endl;
    for (const auto& image_path : class_1_images) {
        char nom[250];
        strcpy(nom, image_path.c_str());
        std::vector<float> probabilities = test(nom);
        
        // Prédiction : classe avec la plus haute probabilité
        int predicted_class = (probabilities[0] > probabilities[1]) ? 0 : 1;
        int true_class = 1;
        
        // Mise à jour des métriques
        if (true_class == 1 && predicted_class == 1) {
            TP++; // Vrai Positif
        } else if (true_class == 1 && predicted_class == 0) {
            FN++; // Faux Négatif
        }
        
        std::cout << "Image: " << image_path << " - Vraie classe: " << true_class 
                  << ", Prédite: " << predicted_class << " (P0: " << probabilities[0] 
                  << ", P1: " << probabilities[1] << ")" << std::endl;
    }
    
    // Calcul des métriques
    std::cout << "\n========== MÉTRIQUES DE CLASSIFICATION ==========" << std::endl;
    std::cout << "Vrais Positifs (TP): " << TP << std::endl;
    std::cout << "Vrais Négatifs (TN): " << TN << std::endl;
    std::cout << "Faux Positifs (FP): " << FP << std::endl;
    std::cout << "Faux Négatifs (FN): " << FN << std::endl;
    
    // Précision, Rappel et F1-Score
    float precision = (TP + FP > 0) ? (float)TP / (TP + FP) : 0.0f;
    float recall = (TP + FN > 0) ? (float)TP / (TP + FN) : 0.0f;
    float f1_score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0f;
    float accuracy = (float)(TP + TN) / (TP + TN + FP + FN);
    
    std::cout << "\nPrécision: " << precision << std::endl;
    std::cout << "Rappel (Sensibilité): " << recall << std::endl;
    std::cout << "F1-Score: " << f1_score << std::endl;
    std::cout << "Exactitude (Accuracy): " << accuracy << std::endl;
    std::cout << "=================================================" << std::endl;
}


 std::vector<OCTET*> couche1(std::vector<OCTET*> images){
    // faire pour chaque images
    // filtre bas * 2
    // filtre haut * 2
    // filtre coefficients random

    // reLU

    // pooling
 }