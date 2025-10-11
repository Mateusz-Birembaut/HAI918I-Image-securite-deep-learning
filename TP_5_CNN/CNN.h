#include "../src/image_ppm.h"
#include "../src/codage.h"

#include <filesystem>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>

std::mt19937 gen(20);

float PSNR_PGM_FLOAT(char cNomImgLue[250], char cNomImgLue2[250])
{
    int nH, nW, nTaille;
    OCTET *ImgOriginal, *ImgModifiee;

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgOriginal, OCTET, nTaille);
    lire_image_pgm(cNomImgLue, ImgOriginal, nTaille);

    allocation_tableau(ImgModifiee, OCTET, nTaille);
    lire_image_pgm(cNomImgLue2, ImgModifiee, nTaille);

    float eqm = 0.0;
    for (int i = 0; i < nTaille; i++)
    {
        int diff = ImgOriginal[i] - ImgModifiee[i];
        eqm += diff * diff;
    }

    free(ImgOriginal);
    free(ImgModifiee);

    if (eqm < 1e-10)
    {                 
        return 100.0f; 
    }

    eqm /= nTaille;

    float psnr = 10.0f * log10(pow(255, 2) / eqm);
    return psnr;
}


OCTET *calculer_image_moyenne(const std::vector<std::vector<OCTET>> &imagettes)
{
    // 1. Vérifications de sécurité
    if (imagettes.empty())
    {
        std::cerr << "Erreur : Le vecteur d'imagettes est vide." << std::endl;
        return nullptr;
    }

    size_t taille_imagette = imagettes[0].size();
    for (const auto &img : imagettes)
    {
        if (img.size() != taille_imagette)
        {
            std::cerr << "Erreur : Toutes les imagettes n'ont pas la même taille." << std::endl;
            return nullptr;
        }
    }

    std::vector<long> sommes_pixels(taille_imagette, 0L);

    for (const auto &img : imagettes)
    {
        for (size_t i = 0; i < taille_imagette; ++i)
        {
            sommes_pixels[i] += img[i];
        }
    }

    OCTET *image_moyenne;
    allocation_tableau(image_moyenne, OCTET, taille_imagette);

    long nb_imagettes = imagettes.size();
    for (size_t i = 0; i < taille_imagette; ++i)
    {
        int moyenne = static_cast<int>(sommes_pixels[i] / nb_imagettes);
        image_moyenne[i] = clamp(moyenne);
    }

    return image_moyenne;
}


OCTET *vector_to_array(const std::vector<OCTET> &vec_in)
{
    OCTET *array_out;
    allocation_tableau(array_out, OCTET, vec_in.size());

    std::copy(vec_in.begin(), vec_in.end(), array_out);

    return array_out;
}

inline OCTET clamp(int value, int min = 0, int max = 255)
{
    return value < min ? min : (value > max ? max : value);
}

void relu(float *image, int nW, int nH)
{
    for (int i = 0; i < nH; i++)
    {
        for (int j = 0; j < nW; j++)
        {
            image[i * nW + j] = std::max(0.0f, image[i * nW + j]);
        }
    }
}

struct Kernel
{

    Kernel(int _size, std::vector<float> &&_values)
        : size(_size), values(std::move(_values))
    {
        if (values.size() != size * size)
        {
            throw std::invalid_argument("Le nombre de valeurs ne correspond pas à la taille du noyau.");
        }
    }

    float sumValues()
    {
        float sum{};
        for (auto val : values)
        {
            sum += val;
        }
        return sum;
    }

    int size;
    std::vector<float> values;
};

struct Layer
{
    std::vector<std::vector<float>> weightsInputs;
    std::vector<float> values;
    std::vector<float> proba;

    Layer(const std::vector<float> &inputs, int nbOfNodes)
    {
        values.resize(nbOfNodes);
        proba.resize(nbOfNodes);
        weightsInputs.resize(inputs.size());


        std::uniform_real_distribution<float> dis(-0.05f, 0.05f);

        for (int i{0}; i < inputs.size(); ++i)
        {
            weightsInputs[i].resize(nbOfNodes);
            for (int j{0}; j < nbOfNodes; j++)
            {
                weightsInputs[i][j] = dis(gen); //1.0f/(nbOfNodes * inputs.size());//// 
            }
        }

        computeWeights(inputs);
    }

    void computeWeights(const std::vector<float> &inputs)
    {
        for (int i{0}; i < values.size(); i++)
        {
            float sum{};
            for (int j{0}; j < inputs.size(); j++)
            {
                sum += weightsInputs[j][i] * inputs[j];
            }
            values[i] = sum;
        }
    }

    float sumValues()
    {
        float sum{};
        for (int i{0}; i < values.size(); i++)
        {
            sum += exp(values[i]);
        }
        return sum;
    }

    void softmax()
    {
        float sumExpInv{1.0f / sumValues()};

        for (int i{0}; i < proba.size(); i++)
        {
            proba[i] = exp(values[i]) * sumExpInv;
        }
    }
};

using FilterFunction = float (*)(float *, int, int, int, int, Kernel);

float convolution_function(float *image, int nW, int nH, int ligne, int colonne, Kernel kernel)
{
    float sum = 0.0f;

    if (ligne < kernel.size / 2 || colonne < kernel.size / 2 || colonne >= nW - kernel.size / 2 || ligne >= nH - kernel.size / 2)
    {
        return 0;
    }

    for (int i = -kernel.size / 2; i <= kernel.size / 2; i++)
    {
        for (int j = -kernel.size / 2; j <= kernel.size / 2; j++)
        {
            int voisinLigne = ligne + i;
            int voisinColonne = colonne + j;

            if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW)
            {
                int kernelIndex = (i + kernel.size / 2) * kernel.size + (j + kernel.size / 2);
                sum += image[voisinLigne * nW + voisinColonne] * kernel.values[kernelIndex];
            }
        }
    }
    return sum;
}

float *convolution(float *pt_image, int nH, int nW, Kernel kernel, FilterFunction filter)
{
    int nTaille;

    float *ImgOut;

    nTaille = nH * nW;
    allocation_tableau(ImgOut, float, nTaille);

    for (int i = 0; i < nH; i++)
    {
        for (int j = 0; j < nW; j++)
        {
            ImgOut[i * nW + j] = filter(pt_image, nW, nH, i, j, kernel);
        }
    }

    return ImgOut;
}

float pool_function(float *image, int nW, int nH, int ligne, int colonne, Kernel kernel)
{
    float max = image[ligne * nW + colonne];
    for (int i = 0; i < kernel.size; i++)
    {
        for (int j = 0; j < kernel.size; j++)
        {
            int voisinLigne = ligne + i;
            int voisinColonne = colonne + j;

            if (voisinLigne >= 0 && voisinLigne < nH && voisinColonne >= 0 && voisinColonne < nW)
            {
                auto value = image[voisinLigne * nW + voisinColonne];
                if (value > max)
                    max = value;
            }
        }
    }
    return max;
}

void pooling(float *&ImgIn, int nH, int nW, Kernel kernel, FilterFunction filter)
{
    float *ImgOut;

    int nH2 = (nH / kernel.size) - 1;
    int nW2 = (nW / kernel.size) - 1;

    int nTaille = nH2 * nW2;
    allocation_tableau(ImgOut, float, nTaille);

    for (int i = 1; i < nH2 - 1; i++)
    {
        for (int j = 1; j < nW2 - 1; j++)
        {
            ImgOut[i * nW2 + j] = filter(ImgIn, nH, nW, i * kernel.size, j * kernel.size, kernel);
        }
    }

    ImgIn = ImgOut;
    // fuite mémoire ici mais pas grave pour l'instant
}

std::vector<float> test(char nom[250], std::vector<std::vector<OCTET>> &imagettes)
{
    int nH, nW, nTaille;

    std::vector<Kernel> kernels_pass1 = {
        Kernel(3, std::vector<float>{
                      1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                      1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                      1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f}),
        Kernel(3, std::vector<float>{1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f, 2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}),
        Kernel(3, std::vector<float>{0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f}), Kernel(3, std::vector<float>{-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f}),
        Kernel(3, std::vector<float>{-1.0f, -5.0f, 4.0f, 2.0f, 3.0f, -4.0f, -6.0f, 1.0f, 5.0f})};

    std::vector<float *> images(kernels_pass1.size());

    FilterFunction filter = &convolution_function;

    float *ImgInC1;

    lire_nb_lignes_colonnes_image_pgm(nom, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgInC1, float, nTaille);

    lire_image_pgm_float(nom, ImgInC1, nTaille);

    for (int j = 0; j < kernels_pass1.size(); j++)
    {

        float *ImgOutC1;

        allocation_tableau(ImgOutC1, float, nTaille);

        ImgOutC1 = convolution(ImgInC1, nH, nW, kernels_pass1[j], filter);

        relu(ImgOutC1, nH, nW);

        images[j] = ImgOutC1;
    }

    FilterFunction pool_fun = &pool_function;
    Kernel poolKernel{2, std::vector<float>{1, 1, 1, 1}};

    for (int i{0}; i < images.size(); ++i)
    {
        pooling(images[i], nH, nW, poolKernel, pool_fun);
    }

    nH /= poolKernel.size;
    nW /= poolKernel.size;
    nH--;
    nW--;


    nTaille = nH * nW;

    // -------------------------PASS 2-----------------
    std::vector<Kernel> kernels_pass2 = {
        Kernel(3, std::vector<float>{
                      1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                      1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                      1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f}),
        Kernel(3, std::vector<float>{0.0f, -1.0f, 0.0f, -1.0f, 5.0f, -1.0f, 0.0f, -1.0f, 0.0f}),
        Kernel(3, std::vector<float>{-1.0f, -5.0f, 4.0f, 2.0f, 3.0f, -4.0f, -6.0f, 1.0f, 5.0f})};

    std::vector<float *> images_pass2(images.size() * kernels_pass2.size());

    int outputIndex = 0;
    for (int i{0}; i < images.size(); ++i)
    {
        for (int j{0}; j < kernels_pass2.size(); j++)
        {

            float *ImgOut;

            allocation_tableau(ImgOut, float, nTaille);

            ImgOut = convolution(images[i], nH, nW, kernels_pass2[j], filter);

            relu(ImgOut, nH, nW);

            images_pass2[outputIndex] = ImgOut;
            outputIndex++;
        }
        free(images[i]);
    }

    for (int i{0}; i < images_pass2.size(); ++i)
    {
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
    for (int i{0}; i < images_pass2.size(); ++i)
    {
        for (int j{0}; j < nTaille; ++j)
        {
            data.push_back(images_pass2[i][j]);
        }
        free(images_pass2[i]);
    }


    auto min_max_it = std::minmax_element(data.begin(), data.end());
    float min_val = *min_max_it.first;
    float max_val = *min_max_it.second;
    float range = max_val - min_val;

    std::vector<OCTET> image(data.size());

    for (int i{0}; i < static_cast<int>(data.size()); ++i)
    {
        float normalized_value = (data[i] - min_val) / range;
        image[i] = static_cast<OCTET>(normalized_value * 255.0f);
    }

    imagettes.push_back(image);


    Layer finalLayer(data, 2);
    finalLayer.softmax();

    return finalLayer.proba;
}


void grade2() {
    // === 1. CHARGEMENT ET SÉPARATION DES DONNÉES ===
    std::string base_path = "images/";
    std::vector<std::string> all_class_0_images, all_class_1_images;
    
    // charger les chemins des 100 images de la classe 0
    std::string class_0_path = base_path + "class_0";
    if (std::filesystem::exists(class_0_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(class_0_path)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".pgm" || entry.path().extension() == ".ppm")) {
                all_class_0_images.push_back(entry.path().string());
            }
        }
    }
    
    // charger les chemins des 100 images de la classe 1
    std::string class_1_path = base_path + "class_1";
    if (std::filesystem::exists(class_1_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(class_1_path)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".pgm" || entry.path().extension() == ".ppm")) {
                all_class_1_images.push_back(entry.path().string());
            }
        }
    }
    
    // assure qu'on a assez d'images
    if (all_class_0_images.size() < 100 || all_class_1_images.size() < 100) {
        std::cerr << "Erreur : Il faut au moins 100 images dans chaque classe." << std::endl;
        return;
    }
    
    std::shuffle(all_class_0_images.begin(), all_class_0_images.end(), gen);
    std::shuffle(all_class_1_images.begin(), all_class_1_images.end(), gen);

    std::vector<std::string> train_c0(all_class_0_images.begin(), all_class_0_images.begin() + 80);
    std::vector<std::string> test_c0(all_class_0_images.begin() + 80, all_class_0_images.begin() + 100);
    
    std::vector<std::string> train_c1(all_class_1_images.begin(), all_class_1_images.begin() + 80);
    std::vector<std::string> test_c1(all_class_1_images.begin() + 80, all_class_1_images.begin() + 100);

    // === 2. CALCUL DES IMAGETTES MOYENNES (sur les 80 images d'entraînement) ===
    std::vector<std::vector<OCTET>> imagettes_train_C0, imagettes_train_C1;

    std::cout << "Génération des imagettes pour les 80 images d'entraînement C0..." << std::endl;
    for (const auto& image_path : train_c0) {
        test((char*)image_path.c_str(), imagettes_train_C0);
    }
    std::cout << "Génération des imagettes pour les 80 images d'entraînement C1..." << std::endl;
    for (const auto& image_path : train_c1) {
        test((char*)image_path.c_str(), imagettes_train_C1);
    }

    OCTET* image_moyenne_c0 = calculer_image_moyenne(imagettes_train_C0);
    OCTET* image_moyenne_c1 = calculer_image_moyenne(imagettes_train_C1);

    // Sauvegarder les moyennes pour les utiliser avec PSNR_PGM
    ecrire_image_pgm((char*)"moyenne_c0.pgm", image_moyenne_c0, 27, 20);
    ecrire_image_pgm((char*)"moyenne_c1.pgm", image_moyenne_c1, 27, 20);
    std::cout << "Images moyennes sauvegardées." << std::endl;

    // === 3. CALCUL DES PSNR (sur les 20 images de test) ===
    float total_psnr_c0_vs_moyenne_c0 = 0.0f;
    float total_psnr_c0_vs_moyenne_c1 = 0.0f;
    
    std::cout << "\nCalcul des PSNR pour les 20 images de test de la classe 0..." << std::endl;
    for (const auto& image_path : test_c0) {
        std::vector<std::vector<OCTET>> temp_imagette_vec;
        test((char*)image_path.c_str(), temp_imagette_vec);
        ecrire_image_pgm((char*)"temp_imagette.pgm", temp_imagette_vec[0].data(), 27, 20);

        total_psnr_c0_vs_moyenne_c0 += PSNR_PGM_FLOAT((char*)"temp_imagette.pgm", (char*)"moyenne_c0.pgm");
        total_psnr_c0_vs_moyenne_c1 += PSNR_PGM_FLOAT((char*)"temp_imagette.pgm", (char*)"moyenne_c1.pgm");
    }

    float total_psnr_c1_vs_moyenne_c0 = 0.0f;
    float total_psnr_c1_vs_moyenne_c1 = 0.0f;

    std::cout << "Calcul des PSNR pour les 20 images de test de la classe 1..." << std::endl;
    for (const auto& image_path : test_c1) {
        std::vector<std::vector<OCTET>> temp_imagette_vec;
        test((char*)image_path.c_str(), temp_imagette_vec);
        ecrire_image_pgm((char*)"temp_imagette.pgm", temp_imagette_vec[0].data(), 27, 20);

        total_psnr_c1_vs_moyenne_c0 += PSNR_PGM_FLOAT((char*)"temp_imagette.pgm", (char*)"moyenne_c0.pgm");
        total_psnr_c1_vs_moyenne_c1 += PSNR_PGM_FLOAT((char*)"temp_imagette.pgm", (char*)"moyenne_c1.pgm");
    }

    // === 4. AFFICHAGE DES RÉSULTATS ===
    std::cout << "\n========== RÉSULTATS PSNR MOYENS ==========" << std::endl;
    std::cout << "PSNR moyen (Test C0 vs Moyenne C0): " << total_psnr_c0_vs_moyenne_c0 / test_c0.size() << " dB" << std::endl;
    std::cout << "PSNR moyen (Test C0 vs Moyenne C1): " << total_psnr_c0_vs_moyenne_c1 / test_c0.size() << " dB" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "PSNR moyen (Test C1 vs Moyenne C0): " << total_psnr_c1_vs_moyenne_c0 / test_c1.size() << " dB" << std::endl;
    std::cout << "PSNR moyen (Test C1 vs Moyenne C1): " << total_psnr_c1_vs_moyenne_c1 / test_c1.size() << " dB" << std::endl;
    std::cout << "=============================================" << std::endl;

    ecrire_image_pgm("moyenne C0.pgm", image_moyenne_c0, 20, 27);
    ecrire_image_pgm("moyenne C1.pgm", image_moyenne_c1, 20, 27);
    // === 5. NETTOYAGE ===
    free(image_moyenne_c0);
    free(image_moyenne_c1);
}

void grade()
{
    std::string base_path = "images/";
    std::vector<std::string> class_0_images;
    std::vector<std::string> class_1_images;

    // charger les chemins des 100 images de la classe 0
    std::string class_0_path = base_path + "class_0";
    if (std::filesystem::exists(class_0_path))
    {
        for (const auto &entry : std::filesystem::directory_iterator(class_0_path))
        {
            if (entry.is_regular_file())
            {
                std::string extension = entry.path().extension();
                if (extension == ".pgm" || extension == ".ppm")
                {
                    class_0_images.push_back(entry.path().string());
                }
            }
        }
    }

    // charger les chemins des 100 images de la classe 1
    std::string class_1_path = base_path + "class_1";
    if (std::filesystem::exists(class_1_path))
    {
        for (const auto &entry : std::filesystem::directory_iterator(class_1_path))
        {
            if (entry.is_regular_file())
            {
                std::string extension = entry.path().extension();
                if (extension == ".pgm" || extension == ".ppm")
                {
                    class_1_images.push_back(entry.path().string());
                }
            }
        }
    }

    std::shuffle(class_0_images.begin(), class_0_images.end(), gen);
    std::shuffle(class_1_images.begin(), class_1_images.end(), gen);

    if (class_0_images.size() > 80)
    {
        class_0_images.resize(80);
    }
    if (class_1_images.size() > 80)
    {
        class_1_images.resize(80);
    }

    std::cout << "Classe 0: " << class_0_images.size() << " images trouvées" << std::endl;
    std::cout << "Classe 1: " << class_1_images.size() << " images trouvées" << std::endl;

    int TP = 0, TN = 0, FP = 0, FN = 0;

    std::vector<std::vector<OCTET>> imagettes_C0;
    std::vector<std::vector<OCTET>> imagettes_C1;

    std::cout << "Traitement des images de la classe 0..." << std::endl;
    for (const auto &image_path : class_0_images)
    {
        char nom[250];
        strcpy(nom, image_path.c_str());
        std::vector<float> probabilities = test(nom, imagettes_C0);

        int predicted_class = (probabilities[0] > probabilities[1]) ? 0 : 1;
        int true_class = 0;

        if (true_class == 0 && predicted_class == 0)
        {
            TN++; 
        }
        else if (true_class == 0 && predicted_class == 1)
        {
            FP++; 
        }

    }

    std::cout << "Traitement des images de la classe 1..." << std::endl;
    for (const auto &image_path : class_1_images)
    {
        char nom[250];
        strcpy(nom, image_path.c_str());
        std::vector<float> probabilities = test(nom, imagettes_C1);

        int predicted_class = (probabilities[0] > probabilities[1]) ? 0 : 1;
        int true_class = 1;

        if (true_class == 1 && predicted_class == 1)
        {
            TP++; 
        }
        else if (true_class == 1 && predicted_class == 0)
        {
            FN++; 
        }

    }

    std::cout << "\n========== MÉTRIQUES DE CLASSIFICATION ==========" << std::endl;
    std::cout << "Vrais Positifs (TP): " << TP << std::endl;
    std::cout << "Vrais Négatifs (TN): " << TN << std::endl;
    std::cout << "Faux Positifs (FP): " << FP << std::endl;
    std::cout << "Faux Négatifs (FN): " << FN << std::endl;

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

