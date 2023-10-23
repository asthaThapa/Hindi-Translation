import pandas as pd

# Create a DataFrame with mixed data types
data = {
    'Hindi_Sentence': ['मेरा नाम जॉन है', 'कैसे हो आप ?', 'बहुत धन्यवाद'],
    'English_Word': ['apple मेरा', 'banana जॉन', 'cherry'],
    'Numeric_Value': [42, 3.14, 100],
    'Boolean_Value': [True, False, True],
    'Mixed_Data': [('प्याज', 'onion'), ('टमाटर', 'tomato'), ('आलू', 'potato')],
    'Dictionary_Data': [{'Name': 'टमाटर', 'Age': 30}, {'Name': 'आलू', 'Age': 25}, {'Name': 'Bob', 'Age': 35}]
}

df = pd.DataFrame(data)

# Write the DataFrame to a Parquet file
df.to_parquet('mixed_data.parquet')

# Create a DataFrame with mixed data types
data_2 = {
    'Hindi_Sentence': ['मुझे दर्द हो रहा है', 'मुझे फुटबॉल खेलना पसंद है', 'बढ़िया काम, Henry'],
    'English_Word': ['basketball घर', 'chair बढ़िया', 'TV'],
    'Numeric_Value': [42, 3.14, 100],
    'Boolean_Value': [True, False, True],
    'Mixed_Data': [('जूता', 'Shoes'), ('घर', 'House'), ('फुटबॉल', 'Football')],
    'Dictionary_Data': [{'Name': 'टमाटर', 'Age': 30}, {'Name': 'आलू', 'Age': 25}, {'Name': 'Bob', 'Age': 35}]
}

df = pd.DataFrame(data_2)

df.to_parquet('mixed_data_2.parquet')

