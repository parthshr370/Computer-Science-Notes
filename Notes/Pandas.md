# Pandas 

1. Python Library used to work with relational or labelled data 
2. This library is built over the numpy library 
3. To install pandas we use the `pip` command 
```python 
pip install pandas
```
Or for arch based systems 
```python
sudo pacman -S python-pandas
```

4. Pandas uses two ways to store data - **Series** and **dataframe** 
5. **Series** is a 1D array that can store datatypes (int, string, float, objects). Basically a column in excel sheet
6. To create a series in Pandas 
```python
import pandas as pd 
import numpy as np

# empty series 
ser = pd.Series()
print("Pandas Series" , ser )

data = np.array(['g' ,'f','e','k'])

ser = pd.Series(data)
print("Pandas :" , ser )
```

7. **Dataframe is a 2D** size mutable tabular data structure with axes(rows and columns) . More like a 2D table 
8. You can create a Dataframe by loading a pre-exisiting dataset in the form of CSV file or Excel 
```python 
import pandas as pd 

df = pd.Dataframe()
print(df)

list1 = ['Kanye','Kendrick','Pinkfloyd']

df = pd.DataFrame(list1)
print(df)
```

9. We can also use index defined and then print it in the data frame 
```python 
import pandas as pd 
data = ['Cant tell me nothing','N95','FLower boy']

index = [1,2,3]

new1 = pd.Series(data,index)
print(new1)
```

10. Or we can just create a Dataframe  with multiple columns 
```python
import pandas as pd
import numpy as np

Dic2 = {
    'Songs':['Bound2','Flashing LIghts', 'N95 '],
    'Artist':['Kanye','Kendrick','Jcole']
}
Dataf = pd.DataFrame(Dic2, index = ['1st ','2nd ', '3rd ']) # this is index that gives out 1nd 2nd 3rd as serial numbers 

print(Dataf)
```

11. We can read CSV files with the help of Pandas 
```python 
import pandas as pd 

dataf = pd.read_csv('Table.csv')
```

12. Now you have loaded the CSV file in the python file , you can print the header and tail of the data 
```python 
#displaying first five rows
display(data_frame.head())
 
#displaying last five rows
display(data_frame.tail())
```

13. To print the column names of the Data frame in a list - 
```python 
print(list(data_frame.columns))
```

14. To get a descriptive Statistical measure of the Data Frame 
```python 
data_frame.describe()

# this will display the central tendancy of the data 
# alog with dispersion and shape of dataset distribution 
```

15. For numeric data, the result’s index will include **count, mean, std, min, and max** as well as lower, 50, and upper percentiles.
![describe](https://media.geeksforgeeks.org/wp-content/uploads/20210414111316/dataframedescribe.png)

16. To find out empty cell in the dataset we can use `isnull` function 
```python 
print(data_frame.isnull()) # tells the empty cells 

print(data_frame.isnull().sum()) # this prints out the number of cells giving out null values 
```