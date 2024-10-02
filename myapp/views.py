from django.shortcuts import render,HttpResponse,redirect
import instaloader
import os
import shutil
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# Create your views here.
def index(request):
    print(request)
    if request.method == 'POST':
        instagram_profile_context = get_instagram_profile(request)
        
    else:
        instagram_profile_context = get_demo_data()
    return render(request, 'index.html', instagram_profile_context)


def get_instagram_profile(request):
    if request.method == 'POST':
        profile_name = request.POST.get('profile_name')

        if profile_name:
            # Initialize Instaloader
            loader = instaloader.Instaloader()

            try:
                # Load profile by username
                profile = instaloader.Profile.from_username(loader.context, profile_name)

                loader.download_profilepic(profile)
               
                files = os.listdir(profile.username)
                current_name = files[0]
                if current_name=="2018-11-21_19-35-46_UTC_profile_pic.jpg":
                    hasprofilepic=0
                else:
                    hasprofilepic=1

                print(hasprofilepic)
                source_path =f"{profile.username}/{current_name}"
                destination_path=f"static/"
                shutil.move(source_path, destination_path)
                
                profilepath =f"../static/{current_name}"
                folder_path =f"{profile.username}"
                print(profilepath)
                shutil.rmtree(folder_path)

                
                modeldata={
                    'profile_name':profile.username,
                    'fullname':profile.full_name,
                    'post_count': profile.mediacount,
                    'follower_count': profile.followers,
                    'following_count': profile.followees,
                    'hasprofilepic':hasprofilepic,
                    'description_length':len(profile.biography),
                    'external_link_present':1 if profile.external_url else 0,
                    'isprivate':1 if profile.is_private else 0,
                    'isverified':0 if profile.is_verified else 0

                }
                print(modeldata)
                model_result=predictor(modeldata)
                if model_result==1:
                    result="fake"
                else:
                    result="Real"

                # Extract profile data
                profile_data = {
                    'profile_name': profile.username,
                    'full_name': profile.full_name,
                    'profile_picture': profile.profile_pic_url,
                    'post_count': profile.mediacount,
                    'follower_count': profile.followers,
                    'following_count': profile.followees,
                    'description': profile.biography,
                    'external_links': profile.external_url,
                    'private_or_public': 'Private' if profile.is_private else 'Public',
                    'verified': 'Verified' if profile.is_verified else 'Not Verified',
                    'path':profilepath,
                    'result':result
                   
                }
               

                return profile_data
            except Exception as e:
                # Handle errors
                print(e)
                # Return an empty dictionary or default values in case of error
                return {
                    'profile_name': 'Error',
                    'full_name': '',
                    'post_count': 0,
                    'follower_count': 0,
                    'following_count': 0,
                    'description': '',
                    'external_links': '',
                    'private_or_public': '',
                    'verified': ''
                }
        else:
            # Return default values if profile_name is not provided
            return {
                'profile_name': '',
                'full_name': '',
                'post_count': 0,
                'follower_count': 0,
                'following_count': 0,
                'description': '',
                'external_links': '',
                'private_or_public': '',
                'verified': ''
            }
    else:
        # Return an empty dictionary if the request method is not POST
        return {}

def get_demo_data():
    # Demo data
    return {
        'profile_name': 'Demo Profile',
        'full_name': 'John Doe',
        'post_count': 100,
        'follower_count': 1000,
        'following_count': 500,
        'description': 'This is a demo profile',
        'external_links': 'https://example.com',
        'private_or_public': 'Public',
        'verified': 'Verified'
    }

def calculate_ratio(username):
    num_numerical_chars = sum(c.isdigit() for c in username)
    length = len(username)
    if length == 0:
        return 0
    ratio = num_numerical_chars / length
    return ratio

def count_words(full_name):
    # Split the full name into individual words based on spaces
    words = full_name.split()
    return len(words)

def are_same1(username, full_name):
    return int(username.lower() == full_name.lower())


def predictor(profile):
    if(profile['isverified']):
        return 1
    numsperuser = calculate_ratio(profile['profile_name'])
    numsperfullname=calculate_ratio(profile['fullname'])
    words_fullname=len(profile['fullname'].split())

    are_same=are_same1(profile['profile_name'],profile['fullname'])
    data=np.array([profile['hasprofilepic'],numsperuser,words_fullname,numsperfullname,are_same,profile['description_length'],profile['external_link_present'],profile['isprivate'],profile['post_count'],profile['follower_count'],profile['following_count']])
    print(data)
    
    
    scaler = StandardScaler()
    scaled_data=scaler.fit_transform(data[:,np.newaxis])
    print(scaled_data)
    # scaler=MinMaxScaler()
    # scaler.fit(reshaped_data)
    # scaled_data =scaler.transform(reshaped_data)

    newdata=np.reshape(scaled_data,(1,-1))

    print(newdata)

    model = load_model('model/trained_model.h5')
    predictions = model.predict(newdata)
    print(predictions)
    binary_prediction = (predictions > 0.7).astype(int)
    print(binary_prediction)
    return  binary_prediction[0, 0]





    

