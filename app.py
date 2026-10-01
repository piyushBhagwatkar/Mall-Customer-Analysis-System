import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Mall Customer Analysis System", page_icon="🛍️", layout="wide")

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
REQUIRED = ["CustomerID", "Gender"] + FEATURES


def level(value, low, high):
    return "Low" if value < low else ("High" if value >= high else "Medium")


st.title("🛍️ Mall Customer Analysis System")
st.caption("Cluster customers by spending behaviour and demographic characteristics using K-Means.")

with st.expander("What is this app doing?"):
    st.write(
        "Customers are grouped using **Age**, **Annual Income** and **Spending Score**. "
        "The features are standardized, then K-Means finds groups of similar customers. "
        "Each group is profiled so marketing teams can target segments differently."
    )

# ---------- Data input ----------
file = st.file_uploader("Upload Mall_Customers.csv", type="csv")
if file is not None:
    df = pd.read_csv(file)
elif os.path.exists("Mall_Customers.csv"):
    df = pd.read_csv("Mall_Customers.csv")
    st.info("No file uploaded, so the bundled Mall_Customers.csv is being used.")
else:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"The CSV is missing required columns: {missing}")
    st.stop()

# ---------- Overview ----------
st.subheader("1. Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Customers", df.shape[0])
c2.metric("Missing values", int(df.isnull().sum().sum()))
c3.metric("Duplicate rows", int(df.duplicated().sum()))
st.dataframe(df.head(10), width="stretch")

df = df.dropna(subset=FEATURES).copy()

# ---------- Clustering ----------
st.subheader("2. Clustering")
k = st.slider("Number of clusters (K)", 2, 10, 5)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, df["Cluster"])
st.metric("Silhouette Score", f"{sil:.3f}")

with st.expander("Elbow Method and Silhouette Score for K = 2 to 10"):
    ks = list(range(2, 11))
    wcss, sils = [], []
    for kk in ks:
        m = KMeans(n_clusters=kk, init="k-means++", random_state=42, n_init=10).fit(X_scaled)
        wcss.append(m.inertia_)
        sils.append(silhouette_score(X_scaled, m.labels_))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(ks, wcss, marker="o"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("K"); ax[0].set_ylabel("WCSS")
    ax[1].plot(ks, sils, marker="o", color="green"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("K")
    st.pyplot(fig)

# ---------- Profile ----------
st.subheader("3. Cluster Profile")
profile = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
).round(2)
names = {
    c: f"{level(r.Avg_Income, 45, 70)} income, {level(r.Avg_Spending, 40, 60)} spending"
    for c, r in profile.iterrows()
}
profile["Segment"] = profile.index.map(names)
df["Segment"] = df["Cluster"].map(names)
st.dataframe(profile, width="stretch")

# ---------- Visuals ----------
st.subheader("4. Visualizations")
left, right = st.columns(2)

with left:
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)",
                    hue="Cluster", palette="viridis", s=70, ax=ax1)
    ax1.set_title("Income vs Spending Score")
    st.pyplot(fig1)

with right:
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Cluster"], palette="viridis", s=70, ax=ax2)
    ax2.set_title("Clusters in PCA Space")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    st.pyplot(fig2)
    st.caption(f"PC1 + PC2 explain {pca.explained_variance_ratio_.sum() * 100:.1f}% of the variance.")

fig3, ax3 = plt.subplots(figsize=(6, 3.5))
sns.countplot(data=df, x="Cluster", hue="Cluster", palette="viridis", legend=False, ax=ax3)
ax3.set_title("Customers per Cluster")
st.pyplot(fig3)

# ---------- Predict ----------
st.subheader("5. Predict a New Customer's Segment")
p1, p2, p3 = st.columns(3)
age = p1.number_input("Age", 18, 100, 30)
income = p2.number_input("Annual Income (k$)", 1, 300, 60)
spend = p3.number_input("Spending Score (1-100)", 1, 100, 50)
if st.button("Predict Segment"):
    row = pd.DataFrame([[age, income, spend]], columns=FEATURES)
    c = int(model.predict(scaler.transform(row))[0])
    st.success(f"Cluster {c}: {names[c]}")

# ---------- Download ----------
st.subheader("6. Download Results")
st.download_button("Download Clustered Data (CSV)", df.to_csv(index=False),
                   "clustered_mall_customers.csv", "text/csv")
