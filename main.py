import pandas as pd


df = pd.read_csv(
    "source_data/gender_diplomacy.csv",
    usecols=["year", "cname_send", "cname_receive", "gender"],
)

df.drop(df[df["gender"] == 99].index, inplace=True)

df.rename(
    columns={"cname_send": "sending_country", "cname_receive": "receiving_country"},
    inplace=True,
)

df["gender"] = df["gender"].map({0: "male", 1: "female"})

grouped_df = (
    df.groupby(["year", "sending_country", "gender"]).size().reset_index(name="count")
)

pivoted_df = grouped_df.pivot_table(
    index=["year", "sending_country"], columns="gender", values="count", fill_value=0
).reset_index()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assuming you already have pivoted_df as in your previous code

# Separate features and target variable
X = pivoted_df.drop(columns=["female"])
y = pivoted_df["female"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define preprocessing steps for numeric and categorical columns
numeric_features = ["year"]
categorical_features = ["sending_country"]

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder())])

# Use ColumnTransformer to apply different preprocessing steps to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Combine the preprocessing step with the regression model
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on future years and specific countries
new_data = pd.DataFrame(
    {"year": [2022], "sending_country": ["United States of America"]}
)

predicted_female_diplomats = model.predict(new_data)
print(predicted_female_diplomats)


# top_sending_countries = df["sending_country"].value_counts().nlargest(5).index
# df_top_countries = df[df["sending_country"].isin(top_sending_countries)]

# sb.pairplot(pivoted_df)
# plt.savefig("output/male_female_diplomats_across_years.png")
# plt.close()

# sb.countplot(x="gender", data=df)
# plt.savefig("output/total_diplomats_sent_gender_since_1968.png")
# plt.close()

# sb.countplot(x="gender", data=df[(df.year >= 2013)])
# plt.savefig("output/total_diplomats_sent_gender_since_2013.png")
# plt.close()

# ax = sb.countplot(data=df_top_countries, x="sending_country", hue="gender")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
# plt.subplots_adjust(bottom=0.3)
# plt.savefig("output/diplomats_sent_country_gender_top_5.png")
# plt.close()
