# Real estate price prediction by Institut Louis Bachelier 

***(15/01/2022) The first version of the reduced_images folder contained a corrupted image for ann_35876173. If you face an issue, you can either remove the problematic image in the folder or download the new version of the reduced_images folder that is available. ***

## Challenge context

Institut Louis Bachelier (ILB) is a sponsored research network in Economics and Finance. It is an association as defined by the law of 1901 and was created in 2008 at the instigation of the Treasury and Caisse des Dépôts et Consignations. Through its activities, it aims to involve academics, public authorities and companies in research projects as well as in scientific events and other forums of exchange. The dozens of research projects hosted at ILBfocus on four societal transitions: environmental, digital, demographic and financial.

The ILB Datalab is a team of data scientists working alongside researchers of the ILB network on applied research projects for both public and private actors of our economic and financial ecosystem. The ILB datalab recently collected an extensive amount of French real estate data and would like to conduct analyses and experiments with it. This challenge is an opportunity to do so.

## Challenge goals

The project is a regression task that deals with real estate price estimation. Estimating housing real estate price is quite a common topic, with an important litterature on estimating prices based on usual data such as: location, surface, land, number of bedrooms, age of the building... The approaches are usually sufficient to estimate the price range but lack precision. However, few have worked to see if adding photos of the asset would bring complementary information, enabling a more precise price estimation.

The objective is thus to work on modelling French housing real estate prices based on usual hierarchical tabular data and, a few photos (between 1 and 6) for each asset and see if it allows better performance than a model trained without the photos.

We will value results interpretability to get a better understanding about the valuable features.

## Data description

The output y represents offered housing real estate prices of French assets in euros.

The input X contains:

    a listing identifier
    the property type (house, apartment, condo, mansion...),
    the location (approximated latitude, approximated longitude, city, postal code, exposition, floor when applicable...),
    the size (living area and land area when applicable),
    the number of rooms, bedrooms, bathrooms...
    energy performance indicators (energy and greenhouse gas emissions)
    the number of photos attached to the listing,
    indicators whether there is a cellar, a balcony, air conditioning...


Alongisde this tabular data, we provide a compressed folder containing 1 to 6 photos per listing (see supplementary files). For each listing in the tabular set of data, the corresponding photos are located in the folder named "ann_XX" where XX corresponds to the listing identifier.

Tabular data have not been prepared upstream. We would like to highlight the fact that images can improve or correct some tabular features. The detailed data dictionary is available below:

Name (Type) 	Description 	Compulsory field on web platform
id_annonce (str) 	unique listing identification code 	True
price (float) 	price at which the property is listed 	True
property_type (str) 	property type (house, appatement…) 	True
approximate_latitude (float) 	latitude of the proporty with a small random gaussian added for the sake of anomysisation 	False
approximate_longitude (float) 	longitude of the proporty with a small random gaussian added for the sake of anomysisation 	False
city (str) 	city in which the property is located 	True
postal_code (int) 	postal code of the property 	True
size (float) 	living area of the property 	False
floor (float) 	floor at which the property is located 	False
land_size (float) 	size of the land that comes with the property 	False
energy_performance_value (float) 	energy performance value in kWh/m²/year 	False
energy_performance_category (char) 	energy performance category as defined by the Frenche DPE reglementation: https://www.ecologie.gouv.fr/diagnostic-performance-energetique-dpe 	False
ghg_value (float) 	greenhouse gas emission performance value in kg eqCO2/m² 	False
ghg_category (char) 	greenhouse gas emission performance category as defined by the Frenche DPE reglementation: https://www.ecologie.gouv.fr/diagnostic-performance-energetique-dpe 	False
exposition (str) 	direction the property is facing 	False
nb_rooms (int) 	number rooms in the property 	False
nb_bedrooms (int) 	number bedrooms in the property 	False
nb_bathrooms (int) 	number bathrooms in the property 	False
nb_parking_places (int) 	number parking places coming with the property 	False
nb_boxes (int) 	number of boxes coming with the property 	False
nb_photos (int) 	number of photos posted on the listing 	False
has_a_balcony (binary) 	indicator whether there is a balcony in the property (1 if true, 0 if false) 	False
nb_terraces (binary) 	number terraces in the property 	False
has_a_cellar (binary) 	indicator whether there is a cellar in the property (1 if true, 0 if false) 	False
has_a_garage (binary) 	indicator whether there is a garage in the property (1 if true, 0 if false) 	False
has_air_conditioning (binary) 	indicator whether there is air conditionning in the property (1 if true, 0 if false) 	False
last_floor (binary) 	indicator whether the property is located on the top floor (1 if true, 0 if false) 	False
upper_floors (binary) 	indicator whether the property is located in the upper floors (1 if true, 0 if false) 	False


The global sample size is around 50K listings and 300K photos. The tabular data set take up ~10MB in CSV format, the images take up ~30GB.

We performed a 80%/20% train/test split, that gives us ~40K listings for training and ~10K for testing.

## Benchmark description

Our benchmark consists in an XGBoost regression model taking as input the tabular features from the X_train.csv file alongside a simple embedding of the corresponding listing images.

The tabular features were preprocessed as follows:

    imputing missing categorical features with an 'Unknown' modality
    imputing missing numerical and binary feature with the median value
    simple label encoding of the categorical features


The images were embedded by concatenating the following values:

    average red, green and blue values accross all pixels of the image (dominant color extraction),
    counts, quantiles and mean value of pixel values of a grayscale version of the image,
    counts, quantiles and mean value of the pixel values of the image obtained after applying a sobel edge detection filter on the grayscale version of the images.


For listings with more than one photo, we considered the average of the image vector embeddings described above.

The model was fine tuned with using a randomize parameter search grid and 5 fold cross-validation.
