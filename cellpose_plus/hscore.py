import pandas as pd
import os

# function for class intensity
def classify_intensity(mean_brightness):
    if 147 < mean_brightness <= 191:
        return "weak"
    elif 115 < mean_brightness <= 147:
        return "mean"
    elif 34 <= mean_brightness <= 115:
        return "strong"
    else:
        return "out_of_range"  
# way to the csv
input_csv_path = "cell_brightness_gray.csv"
output_csv_path = "cell_brightness_gray2.csv"

def process_brightness_data(input_csv_path, output_csv_path):
    """
    Process CSV file and add intensity classification column
    """
    try:
        df = pd.read_csv(input_csv_path)
        
        # searching for data
        if 'id' not in df.columns or 'mean_brightness' not in df.columns:
            raise ValueError("In the source file, the connected columns 'id' or 'mean_brightness' are disabled'")
        
        # add class intensity
        df['class_intensity'] = df['mean_brightness'].apply(classify_intensity)
        
        # save
        result_df = df[['id', 'mean_brightness', 'class_intensity']]
        
        # make directory 
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # save in csv
        result_df.to_csv(output_csv_path, index=False)
        
        print(f"File created successfully: {output_csv_path}")
        return result_df
        
    except FileNotFoundError:
        print(f"Error: File {input_csv_path} not found")
        return None
    except Exception as e:
        print(f"An error has occurred: {e}")
        return None

def hscore(input_csv_path, output_csv_path=None):
    """
    Calculate H-score from CSV file with intensity classification
    """
    try:
        # reading from csv
        df = pd.read_csv(input_csv_path)
        
        # searching for columns
        if 'class_intensity' not in df.columns:
            print("Column 'class_intensity' not found. Performing classification...")
            if 'mean_brightness' not in df.columns:
                raise ValueError("Missing column 'mean_brightness'")
            df['class_intensity'] = df['mean_brightness'].apply(classify_intensity)
        
        # count for class intensity
        counts = df['class_intensity'].value_counts()
        total_count = len(df)
        
         
        # counting ratios
        weak_ratio = counts.get('weak', 0) / total_count * 100
        mean_ratio = counts.get('mean', 0) / total_count * 100
        strong_ratio = counts.get('strong', 0) / total_count * 100
        
        # counting H-score
        h_score = 1 * weak_ratio + 2 * mean_ratio + 3 * strong_ratio
        
        # make DataFrame with results
        result_data = {
            'metric': ['H-score', 'weak_percentage', 'mean_percentage', 'strong_percentage'],
            'value': [h_score, weak_ratio, mean_ratio, strong_ratio],
            'count': [total_count, counts.get('weak', 0), counts.get('mean', 0), counts.get('strong', 0)]
        }
        
        result_df = pd.DataFrame(result_data)
        
        # save results
        if output_csv_path:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            result_df.to_csv(output_csv_path, index=False)
            print(f"H-score results are saved in: {output_csv_path}")
        
        print(f"H-score: {h_score:.2f}")
        print(f"Shares: weak={weak_ratio:.2f}%, mean={mean_ratio:.2f}%, strong={strong_ratio:.2f}%")
        
        return h_score
        
    except FileNotFoundError:
        print(f"Error: File {input_csv_path} not found")
        return None
    except Exception as e:
        print(f"An error occurred while calculating the H-score: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # way to the csv
    input_csv_path = "cell_brightness_gray.csv"
    output_csv_path = "cell_brightness_gray2.csv"
    hscore_output_path = "hscore_results.csv"
    
    # Process the data and add classification
    process_brightness_data(input_csv_path, output_csv_path)
    
    # Calculate H-score
    hscore_value = hscore(output_csv_path, hscore_output_path)

