import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import category_encoder as ce

import LinearRegression as lr

wine_reviews = pd.read_csv("wine-reviews/winemag-data_first150k.csv")

for c in wine_reviews.columns:
	print(c);

print("-----------------")

for c in wine_reviews.columns[wine_reviews.isnull().any()].tolist():
	print(c)

wine_reviews = wine_reviews.drop_duplicates(subset=['description','province', 'country'])


print("-----------------")
wine_reviews = wine_reviews.loc[wine_reviews['country'].notnull()]


print("-----------------")
print(wine_reviews.loc[(wine_reviews['region_1'].isnull() == wine_reviews['region_2'].isnull()) == wine_reviews['designation'].isnull()].isnull().sum())


print("-----------------")
print(wine_reviews.isnull().sum())

print("-----------------")
print(wine_reviews.loc[wine_reviews['region_2'].notnull()].groupby('country')['region_1'].nunique())
print("-----------------")

mostCommonRegion = wine_reviews.loc[wine_reviews['region_1'].notnull()].groupby('country').apply(lambda row: row['region_1'].value_counts().index[0]).to_dict()

def fixRegion1(data):
	for i, row in data.iterrows():
		if row["country"] in mostCommonRegion and pd.isnull(row["region_1"]):
			data.set_value(i, "region_1", mostCommonRegion[row["country"]])
	return data


region1Transformer =FunctionTransformer(fixRegion1, validate=False)

#wine_reviews = region1Transformer.transform(wine_reviews)

#print(wine_reviews.isnull().sum())
#print(len(wine_reviews))
#print(wine_reviews["points"].astype(str).str.isnumeric().sum())

#print(wine_reviews['winery'].nunique())

#print('------------------------------')
#print('Simple Linear Regressor')


countryLabelEncoder = LabelEncoder();
provinceLabelEncoder = LabelEncoder();
wineryLabelEncoder = LabelEncoder();
varietyLabelEncoder = LabelEncoder();

wine_reviews['country'] = countryLabelEncoder.fit_transform(wine_reviews.country);
wine_reviews['province'] = provinceLabelEncoder.fit_transform(wine_reviews.province);
wine_reviews['variety'] = varietyLabelEncoder.fit_transform(wine_reviews.variety);
#wine_reviews['winery'] = wineryLabelEncoder.fit_transform(wine_reviews.winery)

features = [
	tf.feature_column.numeric_column('price'),
	tf.feature_column.categorical_column_with_identity('country', len(countryLabelEncoder.classes_)),
	tf.feature_column.categorical_column_with_hash_bucket('designation', 10000),
	tf.feature_column.categorical_column_with_identity('province', len(provinceLabelEncoder.classes_)),
	tf.feature_column.categorical_column_with_hash_bucket('region_1', 1000),
	tf.feature_column.categorical_column_with_hash_bucket('region_2', 1000),
	tf.feature_column.categorical_column_with_identity('variety', len(varietyLabelEncoder.classes_)),
	tf.feature_column.categorical_column_with_hash_bucket('winery', 10000),
]


svm_wine_reviews = wine_reviews.copy();

points = pd.to_numeric(wine_reviews['points'])

wine_reviews = wine_reviews.drop('description', axis=1).drop('points', axis = 1)
wine_reviews = wine_reviews.fillna('', )

wine_reviews['price'] = pd.to_numeric(wine_reviews['price'])
wine_reviews['price'] = wine_reviews['price'].fillna(wine_reviews['price'].mean())

wine_train, wine_test, points_train, points_test = train_test_split(wine_reviews, points, shuffle = True, random_state = 1984, test_size = 0.2)


#with tf.device('device:GPU:1'):
#	simple_linear_regressor = lr.SimpleLinearRegression.construct_standard_regressor(wine_train, points_train, features, 1000)

#	print('Training successfull')
#	results = simple_linear_regressor.evaluate(input_fn = lr.SimpleLinearRegression.construct_input_fn(wine_test, points_test, 1000))

#	for key in sorted(results):
#		print('%s: %s' % (key, results[key]))

regionEncoder = ce.HashingEncoder()

print(svm_wine_reviews['region_1'])

#(wine_reviews['province'].value_counts().head(10) / len(wine_reviews)).plot.bar()
#wine_reviews['points'].value_counts().sort_index().plot.area()
#plt.show()

