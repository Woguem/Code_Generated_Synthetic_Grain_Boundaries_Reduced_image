"""
@author: Dr Yen Fred WOGUEM 

@description: This script generates synthetic reduced images of grain boundaries : 
             grain boundaries with and without noise, noise and vaccum in a material structure.
"""




import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import math
from joblib import Parallel, delayed
import os
import json
import scipy.ndimage as ndi
from datetime import datetime
import itertools


start_time = datetime.now()  # Start time



pixel_size = 0.5

dataset = np.loadtxt('Data_for_genereted_GB_images.txt', skiprows=1) # Download data set containing Nye tensor and the coordinates of atoms

#Positions of atoms
X = dataset[:, 0]
Y = dataset[:, 1]
Z = dataset[:, 2]

#Component of Nye tensor (A)
A11 = dataset[:,3]
A12 = dataset[:,4]
A13 = dataset[:,5]

A21 = dataset[:,6]
A22 = dataset[:,7]
A23 = dataset[:,8]

A31 = dataset[:,9]
A32 = dataset[:,10]
A33 = dataset[:,11]




# Image generation function with dislocations and/or noise

D_x = np.arange(0, 21)
D_y = np.arange(0, 21)

couple_x_y= list(itertools.product(D_x, D_y))

#np.random.shuffle(couple_x_y)

def generate_image(spacing_pixels, num_dislocations, noise_percentage, size_x, size_y, A23, A22, A33, X, Y, couple_x_y, eps, noise=True):

    dx, dy = couple_x_y
    
    A23_total = {}
    A22_total = {}
    A33_total = {}
     
    
    key_name = f"{num_dislocations}_{spacing_pixels}"
    
    A23_total[key_name] = np.zeros((size_x,size_y))
    A22_total[key_name] = np.zeros((size_x,size_y))
    A33_total[key_name] = np.zeros((size_x,size_y))

    max_indices = np.where(A23 == np.max(A23))[0]

    mean_max_index = np.mean(max_indices)

    #print(max_indices, mean_max_index)

    X_max_position = X[int(mean_max_index)]

    Y_max_position = Y[int(mean_max_index)]

    #print(X_max_position, Y_max_position)

    
    X_scaled = X - X_max_position  

    Y_scaled = Y - Y_max_position

    #print(X_scaled, Y_scaled)

    

    for k in range(num_dislocations): 

        
        
        X_pixels = np.round(X_scaled / pixel_size).astype(int)  # Angstrom → pixel conversion
           
        Y_pixels = np.round(Y_scaled / pixel_size).astype(int)  # Angstrom → pixel conversion
            
        translated_X = X_pixels + k * spacing_pixels + dx + eps
        
        translated_Y = Y_pixels + dy + eps  # Keep Y unchanged (you can also move it if necessary)
        
        # Delete out-of-box values while maintaining X-Y correspondence
        valid_indices = [i for i in range(len(translated_X)) if translated_X[i] < size_x]
        translated_X = [translated_X[i] for i in valid_indices]
        translated_Y = [translated_Y[i] for i in valid_indices]  # Identical filtering
        A23 = [A23[i] for i in valid_indices]  # Identical filtering
        A22 = [A22[i] for i in valid_indices]
        A33 = [A33[i] for i in valid_indices]
        
        
        #shift = 20 # Shift the dislocations to the left so that the space between boundary 1 
                   # and the 1st dislocation is roughly equal to the space between the last dislocation and boundary 2.
                    
        #translated_X = [x - shift for x in translated_X]
            
        
        A23_map = np.zeros((size_x,size_y)) # Initialize the matrix map containing the translated dislocation
        A22_map = np.zeros((size_x,size_y))
        A33_map = np.zeros((size_x,size_y))
        
        size = len(translated_X)

        
        
        for l in range(size):
            if 0 <= translated_X[l] < size_x and 0 <= translated_Y[l] < size_y : 
                x_idx = int((translated_X[l]) ) #% size_x)
                y_idx = int((translated_Y[l]) ) #% size_y)
                A23_map[y_idx, x_idx]  =  max(A23_map[y_idx, x_idx], A23[l])
                A22_map[y_idx, x_idx]  =  max(A22_map[y_idx, x_idx], A22[l])
                A33_map[y_idx, x_idx]  =  max(A33_map[y_idx, x_idx], A33[l])
        
        # Add the map of offset dislocations to the total map
        
        A23_total[key_name] += A23_map
        A22_total[key_name] += A22_map
        A33_total[key_name] += A33_map
        
        # Add noise if necessary
        
    if noise:
        num_noise = int(size_x * size_y * noise_percentage)
        noise_x = np.random.randint(0, size_x, num_noise)
        noise_y = np.random.randint(0, size_y, num_noise)
        
        for i in range(num_noise):
            A23_total[key_name][noise_y[i], noise_x[i]] = np.random.normal(0, 0.04)
            A22_total[key_name][noise_y[i], noise_x[i]] = np.random.normal(0, 0.04)
            A33_total[key_name][noise_y[i], noise_x[i]] = np.random.normal(0, 0.04)
        

    

    return A23_total, A22_total, A33_total



# Generate images with specific proportions


size_x = 30 #400   
size_y = 30 #400   


num_images = 1300

spacing_pixels_GB = [20, 21, 22, 23, 24, 25, 26]  #np.random.randint(25, 200, num_images) # Array containing differents Sapcing between dislocations

spacing_pixels_V = [30, 31, 32, 33, 34, 55, 36]

num_dislocations = 3 #np.array([int(np.round(size_x / n)) for n in spacing_pixels]) # Array containing the number of 
                                                                                    # dislocations for each spacing 

noise_percentage = 0.05  # 5% of noise

#print(num_dislocations, spacing_pixels, np.shape(spacing_pixels))        
   

images_23 = {} # Dictionary to collect all images
images_22 = {}
images_33 = {}



for index, (k1, k2) in enumerate(zip(spacing_pixels_GB, spacing_pixels_V)):

    c1 = 0 # Counter
    c2 = 0 # Counter
    c3 = 0 # Counter
    c4 = 0 # Counter
    
    for n_i in range(1, num_images+1, 1):
        
        print(n_i)

        if n_i <= int(0.3*num_images) :  # Create 30% of images with grain boundary dislocations and no other defects
            c1 += 1
            A23_total, A22_total, A33_total = generate_image(
                spacing_pixels=spacing_pixels_GB[index], 
                num_dislocations=num_dislocations, 
                noise_percentage=noise_percentage,
                size_x=size_x, 
                size_y=size_y, 
                A23=A23,
                A22=A22,
                A33=A33, 
                X=X, 
                Y=Y,
                couple_x_y=couple_x_y[c1],
                eps = 0,
                noise=False)
            
            for key_1a, key_1b, key_1c  in zip(A23_total.keys(), A22_total.keys(), A33_total.keys()):
                image_1a = A23_total[key_1a]
                key_name_1a = f"{key_1a}_GB_system_{c1}_{k1}_1a"
                images_23[key_name_1a] = image_1a  # Add image with key

                image_1b = A22_total[key_1b]
                key_name_1b = f"{key_1b}_GB_system_{c1}_{k1}_1b"
                images_22[key_name_1b] = image_1b

                image_1c = A33_total[key_1c]
                key_name_1c = f"{key_1a}_GB_system_{c1}_{k1}_1c"
                images_33[key_name_1c] = image_1c
            
            print('GB')    
                
        elif int(0.3*num_images) < n_i <= int(0.5*num_images) :  # Create 20% of images without defects (vacuum)
            c2 += 1 
            A23_total, A22_total, A33_total = generate_image(
                spacing_pixels=spacing_pixels_V[index], 
                num_dislocations=num_dislocations, 
                noise_percentage=noise_percentage,
                size_x=size_x, 
                size_y=size_y, 
                A23=A23, 
                A22=A22,
                A33=A33,
                X=X, 
                Y=Y,
                couple_x_y=(0, 0), 
                eps = 5,
                noise=False)
            
            for key_2a, key_2b, key_2c in zip(A23_total.keys(), A22_total.keys(), A33_total.keys()):
                image_2a = A23_total[key_2a]
                key_name_2a = f"{key_2a}_Vaccum_{c2}_{k2}_2a"
                images_23[key_name_2a] = image_2a  # Add image with key

                image_2b = A22_total[key_2b]
                key_name_2b = f"{key_2a}_Vaccum_{c2}_{k2}_2b"
                images_22[key_name_2b] = image_2b

                image_2c = A33_total[key_2c]
                key_name_2c = f"{key_2c}_Vaccum_{c2}_{k2}_2c"
                images_33[key_name_2c] = image_2c
            
            #A23_total = np.zeros((size_x, size_y))  # Blank image
            #key_name = f"{0}_{0}_vacuum_system_{c2}"
            #images[key_name] = A23_total  # Add image with key
            
            print('Vaccum')
            
        elif int(0.5*num_images) < n_i <= int(0.7*num_images):  # Create 20% of images with grain boundary dislocations 
                                                                # and with noise
            c3 += 1
            A23_total, A22_total, A33_total = generate_image(
                spacing_pixels=spacing_pixels_GB[index], 
                num_dislocations=num_dislocations, 
                noise_percentage=noise_percentage, 
                size_x=size_x, 
                size_y=size_y, 
                A23=A23,
                A22=A22,
                A33=A33, 
                X=X, 
                Y=Y,
                couple_x_y=couple_x_y[c3],
                eps = 0,
                noise=True)
            for key_3a, key_3b, key_3c in zip(A23_total.keys(), A22_total.keys(), A33_total.keys()):
                image_3a = A23_total[key_3a]
                key_name_3a = f"{key_3a}_GB_noise_system_{c3}_{k1}_3a"
                images_23[key_name_3a] = image_3a  # Add image with key

                image_3b = A22_total[key_3b]
                key_name_3b = f"{key_3b}_GB_noise_system_{c3}_{k1}_3b"
                images_22[key_name_3b] = image_3b

                image_3c = A33_total[key_3c]
                key_name_3c = f"{key_3a}_GB_noise_system_{c3}_{k1}_3c"
                images_33[key_name_3c] = image_3c
                
            
            print('GB_noise')   
            
        else:  # Create 30% of images only with noise 
            c4 += 1

            A23_total, A22_total, A33_total = generate_image(
                spacing_pixels=spacing_pixels_V[index], 
                num_dislocations=num_dislocations, 
                noise_percentage=noise_percentage,
                size_x=size_x, 
                size_y=size_y, 
                A23=A23, 
                A22=A22,
                A33=A33,
                X=X, 
                Y=Y,
                couple_x_y=(0,0),
                eps = 5,
                noise=True)
            
            for key_4a, key_4b, key_4c in zip(A23_total.keys(), A22_total.keys(), A33_total.keys()):
                image_4a = A23_total[key_4a]
                key_name_4a = f"{key_4a}_Noise_{c4}_{k2}_4a"
                images_23[key_name_4a] = image_4a  # Add image with key

                image_4b = A22_total[key_4b]
                key_name_4b = f"{key_4b}_Noise_{c4}_{k2}_4b"
                images_22[key_name_4b] = image_4b

                image_4c = A33_total[key_4c]
                key_name_4c = f"{key_4c}_Noise_{c4}_{k2}_4c"
                images_33[key_name_4c] = image_4c

            #A23_total = np.zeros((size_x, size_y))  # Blank image
            #num_noise = int(size_x * size_y * noise_percentage)
            #noise_x = np.random.randint(0, size_x, num_noise)
            #noise_y = np.random.randint(0, size_y, num_noise)
            #for i in range(num_noise):
            #    A23_total[noise_y[i], noise_x[i]] = np.random.normal(0, 0.06) # Replacing vacuum with noise
            #key_name = f"{0}_{0}_noise_system_{c4}"
            #images[key_name] = A23_total # Add image with key
            
            print('noise')
    
        



def extract_dislocation_cores(images, save_dir, threshold=0.4): 
    """
    Detects all pixels with maximum intensity in each image.
    If the maximum intensity is below the threshold, the image is empty.

    Args:
        images (dict): Dictionary containing generated images { "key_name": np.array }.
        save_dir (str): Directory where JSON files will be saved.
        threshold (float): Minimum threshold for an image to be considered as containing dislocations.
        

    Returns:
            None (writes a JSON file containing annotations).
    """
    annotations = {}
    
    max_dislocation = 0

    

    for i, image in enumerate(images.values()):
        dislocations = []

        # Find the maximum image intensity
        #max_intensity = np.max(image)

        max_intensity_range = np.max(image[0:20 , 10:30])
        
        #print(max_intensity)


        # If the maximum intensity is below the threshold, consider the image as empty 
                # (because there are vacuum images and images with only noise)
        if max_intensity_range == 0:
            dislocations.append({"x": 0, "y": 0, "p1": 0, "p0": 1})
            annotations[f"Image_{i}.png"] = {"dislocations": dislocations}
            

        elif max_intensity_range < threshold and max_intensity_range !=0 :
            dislocations.append({"x": 0, "y": 0, "p1": 0, "p0": 1})
            annotations[f"Image_{i}.png"] = {"dislocations": dislocations}
            
            
        else :

            # Detection of pixels with intensity greater than 0.4 (first tens of max_intensity)
            local_maxima =  image[0:20 , 10:30] >= threshold                                           
        
            # Grouping connected maxima into regions
            labeled_maxima, num_features = ndi.label(local_maxima)
        
            if num_features > max_dislocation : # Sert à récuperer le nombre maximale de dislocations qu'il peut y avoir dans une image
            
                max_dislocation = num_features

            # Find the center of each detected region using the barycenter
            centers = ndi.center_of_mass(image[0:20 , 10:30], labeled_maxima, range(1, num_features + 1))
            centers = np.round(centers).astype(int)  # Convert to integers
            # Adding centers detected as dislocations
            #for y, x in centers:
            #print(centers[0])
            y, x = centers[0]
            dislocations.append({"x": int(x), "y": int(y), "p1": 1, "p0": 0})

            # Save image annotations
            annotations[f"Image_{i}.png"] = {"dislocations": dislocations}

    
    # Create a complete path for the output JSON file
    output_file_1 = os.path.join(save_dir, 'Max_Dislocations.json')
    
    # Save to JSON file in specified location
    with open(output_file_1, "w") as f:
        json.dump({"max_dislocations": max_dislocation}, f, indent=4)

    print(f"Max dislocations saved in {output_file_1}")
    
    # Create a complete path for the output JSON file
    output_file = os.path.join(save_dir, 'Grain_Boundary.json')
    
    # Save to JSON file in specified location
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"Labels saved")


# Define the directory in which to save images and the JSON file

save_dir = "/home/woguem/Bureau/Projet_Machine_Learning/GB_Cu_001_Generation/Generated_Images"   

# Create folder if none exists
os.makedirs(save_dir, exist_ok=True) 

extract_dislocation_cores(images_23, save_dir)  # Label creation




compt_image = -1
    
# Display images generated with different configurations
for key_23, key_22, key_33 in zip(images_23.keys(), images_22.keys(), images_33.keys()):
    
    compt_image += 1
    
    # Create a figure for each image
    fig1, ax1 = plt.subplots(figsize=(13.5, 9), dpi=300)
    
    # Calculating limits
    #extent_angstrong = [0, size_x * pixel_size, 0, size_y * pixel_size]
    
    extent = [0, size_x, 0, size_y]
    
    # Image display with “jet” color map, range of values defined between v_min and v_max
    img_23 = ax1.imshow(images_23[key_23], cmap='gray', origin='lower', extent=extent)
    
    # Add a color bar to the right of the image
    #cbar = fig.colorbar(img, pad=0.1, location='right')
    #cbar.set_label(r'$\alpha_{23}$ [$\mathrm{\AA}^{-1}$]', fontsize=40) 
    #cbar.ax.yaxis.set_label_position('left')
    #cbar.ax.tick_params(labelsize=34)
    
    # Add labels and titles to axes
    #ax.set_xlabel(r'X [$\mathrm{\AA}$]', fontsize=34)
    #ax.set_ylabel(r'Y [$\mathrm{\AA}$]', fontsize=34)
    #ax.tick_params(axis='both', which='major', labelsize=34)

    ax1.set_xlim(10, 30)  
    ax1.set_ylim(0, 20)
    
    ax1.set_axis_off() # Disable axis display
    
    # Add image title
    #ax.set_title(f'Number_of_dislocations_and_Spacing_{key}', fontsize=14, fontweight='bold')
    
    # Save image as PNG
    save_path1 = os.path.join(save_dir, f'Image_{compt_image}_1.png')
    
    fig1.savefig(save_path1, bbox_inches = 'tight', pad_inches = 0, dpi=300)


    fig2, ax2 = plt.subplots(figsize=(13.5, 9), dpi=300)
    extent = [0, size_x, 0, size_y]
    img_22 = ax2.imshow(images_22[key_22], cmap='gray', origin='lower', extent=extent)
    ax2.set_xlim(10, 30)  
    ax2.set_ylim(0, 20)
    ax2.set_axis_off()
    save_path2 = os.path.join(save_dir, f'Image_{compt_image}_2.png')
    
    fig2.savefig(save_path2, bbox_inches = 'tight', pad_inches = 0, dpi=300)


    fig3, ax3 = plt.subplots(figsize=(13.5, 9), dpi=300)
    extent = [0, size_x, 0, size_y]
    img_33 = ax3.imshow(images_33[key_33], cmap='jet', origin='lower', vmin=-0.3, vmax=0.3, extent=extent)
    ax3.set_xlim(10, 30)  
    ax3.set_ylim(0, 20)
    ax3.set_axis_off()
    save_path3 = os.path.join(save_dir, f'Image_{compt_image}_3.png')
    
    fig3.savefig(save_path3, bbox_inches = 'tight', pad_inches = 0, dpi=300)
    
    
    # Display image in viewer window
    #plt.show()
    
    # Close all figures after display
    plt.close() 

    

end_time = datetime.now()  # Fin du chronomètre
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")





































