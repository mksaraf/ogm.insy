"""
Create the chatbot OGM.Insy for life insurance

Author: Geetika Saraf & Manish Kumar Saraf

"""
# Import library

# import warnings
# warnings.filterwarnings('ignore')

# Arguments
#rawDataPath = '/Users/manishsaraf/OneDrive/OGM/OGM-Git/OGM-MLOPS/notebooks/data/bias-in-advertising/ad_campaign_data.csv'


def main():

    # Get raw data
    from steps.getdata import getData
    ad_conversion_dataset = getData(rawDataPath)
    ad_conversion_dataset_output = ad_conversion_dataset.head()
    print(ad_conversion_dataset_output)

    
if __name__ == '__main__':
    main()
