"""
EXP 2 - Data Handling technique with Programming exercise
Dataset: Sub_Division_IMD_2017.csv (IMD subdivision-wise monthly rainfall, India)
Use case: KJS-AGR-01 (AI Irrigation Advisory)  -- same dataset used in EXP1
"""

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

def line(msg=""):
    print("\n" + "="*70)
    print(msg)
    print("="*70)

# ---------------------------------------------------------------
# 1. Read dataset using Numpy and Pandas
# ---------------------------------------------------------------
line("1. Reading dataset")
df = pd.read_csv("Sub_Division_IMD_2017.csv")
print("Loaded with pandas.read_csv(). Shape:", df.shape)

# quick numpy usage - convert one numeric column to a numpy array
annual_arr = np.array(df['ANNUAL'])
print("ANNUAL column as numpy array, dtype:", annual_arr.dtype, "first 5:", annual_arr[:5])

# ---------------------------------------------------------------
# 2 & 3. Detailed data analysis - info(), describe(), shape, columns
# ---------------------------------------------------------------
line("2/3. df.info(), df.describe(), shape, columns")
df.info()
print(df.describe())
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ---------------------------------------------------------------
# 4. Null values - check + count
# ---------------------------------------------------------------
line("4. Null value check")
print(df.isnull().sum())
print("\nTotal nulls in dataset:", df.isnull().sum().sum())

# ---------------------------------------------------------------
# 5. Handle missing values (imputation, using inplace)
# ---------------------------------------------------------------
line("5. Handling missing values")
month_cols = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
season_cols = ['ANNUAL','JF','MAM','JJAS','OND']

# fill month + season rainfall nulls with subdivision-wise mean (rainfall varies a LOT
# by region, so a single global mean would be misleading -> group mean is more sensible)
# note: doing df[col].fillna(..., inplace=True) does NOT work anymore in newer pandas
# (copy-on-write breaks chained assignment) - assigning back to df[col] is the fix.
for col in month_cols + season_cols:
    df[col] = df[col].fillna(df.groupby('SUBDIVISION')[col].transform('mean'))

# a few subdivisions might still have all-NaN for a column in edge years -> fallback to global mean
df.fillna(df[month_cols + season_cols].mean(numeric_only=True), inplace=True)

print("Nulls remaining after imputation:\n", df.isnull().sum().sum())

# ---------------------------------------------------------------
# 6. Sorting + top 5/8 records based on column values (conditional filtering)
# ---------------------------------------------------------------
line("6. Sorting - top records")
print("Top 5 rows by ANNUAL rainfall:")
print(df.sort_values(by='ANNUAL', ascending=False).head(5)[['SUBDIVISION','YEAR','ANNUAL']])

print("\nTop 8 rows by JJAS (monsoon) rainfall, only for years after 2000:")
print(df[df['YEAR'] > 2000].sort_values(by='JJAS', ascending=False).head(8)[['SUBDIVISION','YEAR','JJAS']])

# ---------------------------------------------------------------
# 7. Frequency listing of a column (2 cases)
# ---------------------------------------------------------------
line("7. Frequency counts")
print("Case 1 - record count per SUBDIVISION:")
print(df['SUBDIVISION'].value_counts())

print("\nCase 2 - how many records fall in each rainfall 'bucket' (using YEAR decade instead):")
df['DECADE'] = (df['YEAR'] // 10) * 10
print(df['DECADE'].value_counts().sort_index())

# ---------------------------------------------------------------
# 8. Sorting rows and columns - implicit (iloc) vs explicit (loc) indexing
# ---------------------------------------------------------------
line("8. Implicit vs explicit indexing")
print("Explicit indexing with loc - rows 0 to 4, columns SUBDIVISION to MAR:")
print(df.loc[0:4, 'SUBDIVISION':'MAR'])

print("\nImplicit indexing with iloc - first 5 rows, first 4 columns:")
print(df.iloc[0:5, 0:4])

# sort columns alphabetically just to show column-sorting too
print("\nColumns sorted alphabetically (df.sort_index(axis=1)) - head:")
print(df.sort_index(axis=1).head(2))

# ---------------------------------------------------------------
# 9. Accessing rows based on condition, few columns (3 cases, compound conditions)
# ---------------------------------------------------------------
line("9. Conditional row access (compound conditions)")
print("Case 1 - Konkan & Goa, years after 2010, show YEAR & ANNUAL:")
c1 = df[(df['SUBDIVISION'] == 'Konkan & Goa') & (df['YEAR'] > 2010)]
print(c1[['YEAR','ANNUAL']])

print("\nCase 2 - Rows with ANNUAL > 3000mm AND JJAS (monsoon) > 2000mm:")
c2 = df[(df['ANNUAL'] > 3000) & (df['JJAS'] > 2000)]
print(c2[['SUBDIVISION','YEAR','ANNUAL','JJAS']].head(8))

print("\nCase 3 - Coastal Karnataka OR Kerala, with low rainfall year (ANNUAL < 2000):")
c3 = df[((df['SUBDIVISION'] == 'Coastal Karnataka') | (df['SUBDIVISION'] == 'Kerala')) & (df['ANNUAL'] < 2000)]
print(c3[['SUBDIVISION','YEAR','ANNUAL']])

# ---------------------------------------------------------------
# 10. Min / Max analysis
# ---------------------------------------------------------------
line("10. Min / Max analysis")
print("Max ANNUAL rainfall record:")
print(df.loc[df['ANNUAL'].idxmax(), ['SUBDIVISION','YEAR','ANNUAL']])
print("\nMin ANNUAL rainfall record:")
print(df.loc[df['ANNUAL'].idxmin(), ['SUBDIVISION','YEAR','ANNUAL']])
print("\nUsing numpy directly - np.max/np.min on ANNUAL array:")
print("max:", np.max(df['ANNUAL'].values), " min:", np.min(df['ANNUAL'].values))

# ---------------------------------------------------------------
# 11. Groupby on one or more columns (2 cases)
# ---------------------------------------------------------------
line("11. Groupby")
print("Case 1 - groupby SUBDIVISION, mean ANNUAL rainfall:")
g1 = df.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False)
print(g1.head(10))

print("\nCase 2 - groupby [SUBDIVISION, DECADE], mean JJAS rainfall:")
g2 = df.groupby(['SUBDIVISION','DECADE'])['JJAS'].mean()
print(g2.head(15))

# ---------------------------------------------------------------
# 12. New column derived from existing columns
# ---------------------------------------------------------------
line("12. Derived column")
# irrigation-advisory relevant flag: is monsoon (JJAS) contributing most of annual rainfall?
df['MONSOON_SHARE_PCT'] = (df['JJAS'] / df['ANNUAL']) * 100
print(df[['SUBDIVISION','YEAR','ANNUAL','JJAS','MONSOON_SHARE_PCT']].head())

# ---------------------------------------------------------------
# 13. Aggregate functions with groupby (2 cases)
# ---------------------------------------------------------------
line("13. Aggregate functions with groupby")
print("Case 1 - agg() with multiple stats on ANNUAL, per SUBDIVISION:")
agg1 = df.groupby('SUBDIVISION')['ANNUAL'].agg(['mean','min','max','std'])
print(agg1.head(10))

print("\nCase 2 - agg() on multiple columns at once:")
agg2 = df.groupby('SUBDIVISION').agg({'ANNUAL':'mean', 'JJAS':'mean', 'MONSOON_SHARE_PCT':'mean'})
print(agg2.head(10))

# ---------------------------------------------------------------
# 14. Selection on particular groups (by name / condition)
# ---------------------------------------------------------------
line("14. get_group / selecting specific groups")
grouped = df.groupby('SUBDIVISION')
print("get_group('Konkan & Goa') - head:")
print(grouped.get_group('Konkan & Goa')[['YEAR','ANNUAL']].head())

print("\nSubdivisions whose average ANNUAL rainfall > 3000mm (filter on groupby result):")
high_rain = g1[g1 > 3000]
print(high_rain)

# ---------------------------------------------------------------
# 15. Correlation between two columns
# ---------------------------------------------------------------
line("15. Correlation")
corr_val = df['JJAS'].corr(df['ANNUAL'])
print(f"Correlation between JJAS (monsoon) rainfall and ANNUAL rainfall: {corr_val:.4f}")
print("\nFull correlation matrix of monthly + seasonal columns:")
print(df[month_cols + season_cols].corr().round(2))

# ---------------------------------------------------------------
# 16. Transformation - normalization
# ---------------------------------------------------------------
line("16. Normalization (Min-Max scaling)")
df['ANNUAL_NORM'] = (df['ANNUAL'] - df['ANNUAL'].min()) / (df['ANNUAL'].max() - df['ANNUAL'].min())
print(df[['SUBDIVISION','YEAR','ANNUAL','ANNUAL_NORM']].head())
print("\nCheck range after normalization -> min:", df['ANNUAL_NORM'].min(), " max:", df['ANNUAL_NORM'].max())

# ---------------------------------------------------------------
# 17. Joining, merging, concatenation
# ---------------------------------------------------------------
line("17. Join / Merge / Concat")
# small lookup table: rough state mapping for a few subdivisions (for a merge demo)
state_lookup = pd.DataFrame({
    'SUBDIVISION': ['Konkan & Goa', 'Kerala', 'Coastal Karnataka', 'Gangetic West Bengal'],
    'REGION_TYPE': ['Coastal', 'Coastal', 'Coastal', 'Gangetic Plain']
})
merged = pd.merge(df, state_lookup, on='SUBDIVISION', how='left')
print("Merged dataframe (only matched rows shown):")
print(merged[merged['REGION_TYPE'].notnull()][['SUBDIVISION','YEAR','REGION_TYPE']].head())

# concat demo - stack first 3 and last 3 rows of the dataframe
concat_demo = pd.concat([df.head(3), df.tail(3)])
print("\nConcat demo (first 3 + last 3 rows):")
print(concat_demo[['SUBDIVISION','YEAR','ANNUAL']])

line("Done")
