import reflex as rx
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import Any
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff


class AnalysisState(rx.State):
    """Manages the data analysis pipeline and visualization data."""

    is_loading: bool = True
    df: pd.DataFrame = pd.DataFrame()
    search_query: str = ""
    filter_cluster: str = "All"
    age_min: int = 18
    age_max: int = 75
    income_min: int = 15000
    income_max: int = 200000
    credit_score_min: int = 300
    credit_score_max: int = 850
    products_filter: str = ""

    @rx.event
    def set_search_query(self, value: str):
        self.search_query = value

    @rx.event
    def set_filter_cluster(self, value: str):
        self.filter_cluster = value

    @rx.event
    def set_age_min(self, value: int):
        self.age_min = int(value)

    @rx.event
    def set_age_max(self, value: int):
        self.age_max = int(value)

    @rx.event
    def set_income_min(self, value: int):
        self.income_min = int(value)

    @rx.event
    def set_income_max(self, value: int):
        self.income_max = int(value)

    @rx.event
    def set_credit_score_min(self, value: int):
        self.credit_score_min = int(value)

    @rx.event
    def set_credit_score_max(self, value: int):
        self.credit_score_max = int(value)

    @rx.event
    def set_products_filter(self, value: str):
        self.products_filter = value

    total_customers: int = 0
    quant_vars: list[str] = [
        "age",
        "annual_income",
        "account_balance",
        "credit_score",
        "num_products",
        "tenure_months",
        "monthly_transactions",
        "avg_transaction_amount",
        "loan_amount",
    ]
    pca_variance_data: list[dict] = []
    pca_biplot: go.Figure | None = None
    n_components_80: int = 0
    correlation_heatmap: go.Figure | None = None
    dendrogram_fig: go.Figure | None = None
    cluster_distribution_data: list[dict] = []
    segment_profiles: list[dict] = []
    segment_radar_charts: list[go.Figure] = []
    cluster_mapping: dict[int, str] = {
        0: "Standard",
        1: "Premium",
        2: "Young Starters",
        3: "Budget-Conscious",
    }
    cluster_colors: list[str] = ["#38bdf8", "#fbbf24", "#34d399", "#f87171"]

    # Connected pipeline artifacts
    cleaned_data: pd.DataFrame = pd.DataFrame()
    pca_data: list[list[float]] = []  # serialized PCA coordinates
    clustered_data: list[dict] = []   # rows with cluster labels
    n_clusters: int = 4
    clustering_algorithm: str = "hierarchical"  # or "kmeans"

    @rx.event
    def set_n_clusters(self, value: int):
        self.n_clusters = int(value)

    @rx.event
    def set_clustering_algorithm(self, value: str):
        if value in ("hierarchical", "kmeans"):
            self.clustering_algorithm = value

    @rx.var
    def filtered_customers(self) -> list[dict[str, str | int | float]]:
        """Filters customers based on multiple criteria."""
        if self.df.empty or "cluster" not in self.df.columns:
            return []
        df_copy = self.df.copy()
        df_copy["segment"] = df_copy["cluster"].map(self.cluster_mapping)
        if self.filter_cluster != "All":
            df_copy = df_copy[df_copy["segment"] == self.filter_cluster]
        if self.search_query:
            df_copy = df_copy[
                df_copy["customer_id"].astype(str).str.contains(self.search_query)
            ]
        df_copy = df_copy[df_copy["age"].between(self.age_min, self.age_max)]
        df_copy = df_copy[
            df_copy["annual_income"].between(self.income_min, self.income_max)
        ]
        df_copy = df_copy[
            df_copy["credit_score"].between(
                self.credit_score_min, self.credit_score_max
            )
        ]
        if self.products_filter:
            df_copy = df_copy[df_copy["num_products"] == int(self.products_filter)]
        return df_copy.to_dict("records")

    @rx.var
    def active_filter_count(self) -> int:
        """Counts the number of active filters."""
        count = 0
        if self.filter_cluster != "All":
            count += 1
        if self.age_min != 18 or self.age_max != 75:
            count += 1
        if self.income_min != 15000 or self.income_max != 200000:
            count += 1
        if self.credit_score_min != 300 or self.credit_score_max != 850:
            count += 1
        if self.products_filter:
            count += 1
        return count

    @rx.event
    def clear_all_filters(self):
        """Resets all filter criteria to their default values."""
        self.search_query = ""
        self.filter_cluster = "All"
        self.age_min = 18
        self.age_max = 75
        self.income_min = 15000
        self.income_max = 200000
        self.credit_score_min = 300
        self.credit_score_max = 850
        self.products_filter = ""

    @rx.var
    def marketing_recommendations(self) -> dict[str, dict[str, str | list[str]]]:
        """Provides targeted marketing recommendations for each segment."""
        return {
            "Premium": {
                "title": "Loyalty & Growth",
                "icon": "gem",
                "color": "text-amber-500",
                "description": "High-value customers. Focus on retention, wealth management, and premium product cross-selling.",
                "recommendations": [
                    "Offer exclusive access to wealth management advisors.",
                    "Introduce premium credit cards with high rewards.",
                    "Provide loyalty bonuses and personalized investment opportunities.",
                    "Early access to new financial products.",
                ],
            },
            "Standard": {
                "title": "Engagement & Upselling",
                "icon": "user-check",
                "color": "text-sky-500",
                "description": "Stable customer base. Goal is to increase engagement and introduce new products.",
                "recommendations": [
                    "Promote digital banking features for convenience.",
                    "Offer bundled products (e.g., checking + savings with benefits).",
                    "Cross-sell personal loans or mortgage products.",
                    "Run targeted email campaigns on financial planning.",
                ],
            },
            "Young Starters": {
                "title": "Education & Digital Adoption",
                "icon": "rocket",
                "color": "text-emerald-500",
                "description": "Younger, digitally-savvy customers. Focus on building long-term relationships and financial education.",
                "recommendations": [
                    "Offer mobile-first banking solutions and budgeting tools.",
                    "Provide educational content on savings and investing.",
                    "Promote low-interest starter credit cards.",
                    "Engage through social media and financial wellness workshops.",
                ],
            },
            "Budget-Conscious": {
                "title": "Support & Debt Management",
                "icon": "shield-half",
                "color": "text-red-500",
                "description": "Customers who may need financial support and guidance.",
                "recommendations": [
                    "Offer debt consolidation services or financial counseling.",
                    "Promote high-yield savings accounts to build a safety net.",
                    "Provide tools for expense tracking and budget management.",
                    "Offer fee waivers or lower-cost banking options.",
                ],
            },
        }

    @rx.event
    def download_clustered_data(self):
        """Allows downloading the dataframe with cluster info as a CSV."""
        if not self.df.empty:
            df_to_download = self.df.copy()
            df_to_download["segment"] = df_to_download["cluster"].map(
                self.cluster_mapping
            )
            return rx.download(
                data=df_to_download.to_csv(index=False).encode("utf-8"),
                filename="bank_customers_clustered.csv",
            )

    @rx.event
    def download_chart(self, chart_type: str):
        """Downloads the specified chart as a PNG image."""
        fig = None
        filename = "chart.png"
        if chart_type == "variance" and self.pca_variance_data:
            fig = go.Figure()
            df = pd.DataFrame(self.pca_variance_data)
            fig.add_trace(
                go.Bar(
                    x=df["name"],
                    y=df["variance"],
                    name="Individual Variance",
                    marker_color="#38bdf8",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["name"],
                    y=df["cumulative"],
                    name="Cumulative Variance",
                    yaxis="y2",
                    line=dict(color="#ef4444"),
                )
            )
            fig.update_layout(
                title="Explained Variance per Component",
                yaxis=dict(title="Individual Variance (%)"),
                yaxis2=dict(
                    title="Cumulative Variance (%)", overlaying="y", side="right"
                ),
                barmode="group",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            filename = "variance_explained.png"
        elif chart_type == "biplot" and self.pca_biplot:
            fig = self.pca_biplot
            filename = "pca_biplot.png"
        elif chart_type == "heatmap" and self.correlation_heatmap:
            fig = self.correlation_heatmap
            filename = "correlation_heatmap.png"
        elif chart_type == "dendrogram" and self.dendrogram_fig:
            fig = self.dendrogram_fig
            filename = "dendrogram.png"
        elif chart_type == "distribution" and self.cluster_distribution_data:
            df = pd.DataFrame(self.cluster_distribution_data)
            fig = px.pie(
                df,
                values="value",
                names="name",
                title="Cluster Distribution",
                color_discrete_sequence=self.cluster_colors,
            )
            filename = "cluster_distribution.png"
        if fig:
            img_bytes = fig.to_image(format="png", scale=2)
            return rx.download(data=img_bytes, filename=filename)

    def _create_sample_data(self, file_path: str):
        """Creates and saves a sample banking customer dataset."""
        np.random.seed(42)
        n_customers = 200
        data = {
            "customer_id": range(1, n_customers + 1),
            "age": np.random.randint(18, 75, n_customers),
            "annual_income": np.random.normal(50000, 25000, n_customers).clip(
                15000, 200000
            ),
            "account_balance": np.random.normal(15000, 10000, n_customers).clip(
                0, 100000
            ),
            "credit_score": np.random.randint(300, 850, n_customers),
            "num_products": np.random.randint(1, 6, n_customers),
            "tenure_months": np.random.randint(1, 120, n_customers),
            "monthly_transactions": np.random.randint(5, 100, n_customers),
            "avg_transaction_amount": np.random.normal(200, 150, n_customers).clip(
                10, 2000
            ),
            "loan_amount": np.random.choice(
                [0, 5000, 10000, 25000, 50000],
                n_customers,
                p=[0.4, 0.2, 0.2, 0.15, 0.05],
            ),
            "savings_account": np.random.choice([0, 1], n_customers, p=[0.3, 0.7]),
            "credit_card": np.random.choice([0, 1], n_customers, p=[0.2, 0.8]),
        }
        df = pd.DataFrame(data)
        high_income_mask = df["annual_income"] > 70000
        df.loc[high_income_mask, "account_balance"] += np.random.normal(
            20000, 5000, high_income_mask.sum()
        ).clip(0, 50000)
        df.loc[high_income_mask, "credit_score"] = df.loc[
            high_income_mask, "credit_score"
        ].clip(650, 850)
        young_mask = df["age"] < 30
        df.loc[young_mask, "tenure_months"] = df.loc[young_mask, "tenure_months"].clip(
            1, 36
        )
        df.to_csv(file_path, index=False)
        return df

    @rx.event(background=True)
    async def load_and_analyze_data(self):
        """Loads data, performs cleaning, standardization, PCA, and correlation analysis."""
        async with self:
            self.is_loading = True
        # Prefer cleaned data saved on disk if available.
        clean_file = "assets/bank_customers_clean.csv"
        if os.path.exists(clean_file):
            df = pd.read_csv(clean_file)
        else:
            file_path = "assets/bank_customers.csv"
            if not os.path.exists(file_path):
                df = self._create_sample_data(file_path)
            else:
                df = pd.read_csv(file_path)
        for col in self.quant_vars:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        async with self:
            self.df = df
            self.total_customers = len(df)
        # Persist cleaned dataset in pipeline state
        async with self:
            self.cleaned_data = self.df.copy()

        X = self.df[self.quant_vars].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        variance_explained = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        variance_data = []
        for i, (var, cum_var) in enumerate(
            zip(variance_explained, cumulative_variance)
        ):
            variance_data.append(
                {
                    "name": f"PC{i + 1}",
                    "variance": round(var * 100, 2),
                    "cumulative": round(cum_var * 100, 2),
                }
            )
        n_components_80_val = np.argmax(cumulative_variance >= 0.8) + 1
        loadings = pca.components_
        pc1_loadings = loadings[0]
        pc2_loadings = loadings[1]
        biplot_fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            title="PCA Biplot (PC1 vs PC2)",
            labels={"x": "Principal Component 1", "y": "Principal Component 2"},
        )
        biplot_fig.update_traces(marker=dict(color="#0284c7", opacity=0.7))
        for i, var_name in enumerate(self.quant_vars):
            biplot_fig.add_annotation(
                ax=0,
                ay=0,
                axref="x",
                ayref="y",
                x=pc1_loadings[i] * np.sqrt(pca.explained_variance_[0]) * 3,
                y=pc2_loadings[i] * np.sqrt(pca.explained_variance_[1]) * 3,
                showarrow=True,
                arrowhead=2,
                arrowcolor="#475569",
                font=dict(color="#1e293b", size=10),
            )
            biplot_fig.add_annotation(
                x=pc1_loadings[i] * np.sqrt(pca.explained_variance_[0]) * 3.5,
                y=pc2_loadings[i] * np.sqrt(pca.explained_variance_[1]) * 3.5,
                text=var_name,
                showarrow=False,
                font=dict(color="#334155", size=11, family="Poppins"),
            )
        biplot_fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Poppins, sans-serif"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        corr_matrix = X.corr()
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="Blues",
                colorbar=dict(title="Correlation"),
            )
        )
        heatmap_fig.update_layout(
            title="Variable Correlation Heatmap",
            xaxis_nticks=36,
            font=dict(family="Poppins, sans-serif"),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        # Persist PCA coordinates
        async with self:
            self.pca_data = X_pca[:, :2].tolist()

        linkage_matrix = linkage(X_pca[:, :n_components_80_val], method="ward")
        dendro_fig = ff.create_dendrogram(linkage_matrix, color_threshold=15)
        dendro_fig.update_layout(
            title="Hierarchical Clustering Dendrogram",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Poppins"),
        )
        n_clusters = int(self.n_clusters)
        if self.clustering_algorithm == "kmeans":
            clustering_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = clustering_model.fit_predict(X_pca[:, :n_components_80_val])
        else:
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = clustering_model.fit_predict(X_pca[:, :n_components_80_val])
        async with self:
            self.df["cluster"] = labels
        cluster_counts = self.df["cluster"].value_counts().sort_index()
        dist_data = [
            {
                "name": self.cluster_mapping.get(i, f"Cluster {i}"),
                "value": int(count),
                "fill": self.cluster_colors[i],
            }
            for i, count in cluster_counts.items()
        ]
        profiles = []
        radars = []
        profile_metrics = [
            "annual_income",
            "account_balance",
            "credit_score",
            "monthly_transactions",
            "avg_transaction_amount",
        ]
        for i in range(n_clusters):
            cluster_df = self.df[self.df["cluster"] == i]
            profile = {
                "name": self.cluster_mapping.get(i, f"Cluster {i}"),
                "color": self.cluster_colors[i],
                "size": len(cluster_df),
                "features": cluster_df[self.quant_vars].mean().to_dict(),
            }
            profiles.append(profile)
            radar_data = cluster_df[profile_metrics].mean()
            radar_norm = (radar_data - self.df[profile_metrics].min()) / (
                self.df[profile_metrics].max() - self.df[profile_metrics].min()
            )
            radar_fig = go.Figure()
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=radar_norm,
                    theta=profile_metrics,
                    fill="toself",
                    name=profile["name"],
                    marker=dict(color=self.cluster_colors[i]),
                )
            )
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title=dict(text=f"{profile['name']} Profile"),
                height=300,
                margin=dict(l=40, r=40, t=60, b=40),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Poppins"),
            )
            radars.append(radar_fig)
        async with self:
            self.pca_variance_data = variance_data
            self.pca_biplot = biplot_fig
            self.n_components_80 = int(n_components_80_val)
            self.correlation_heatmap = heatmap_fig
            self.dendrogram_fig = dendro_fig
            self.cluster_distribution_data = dist_data
            self.segment_profiles = profiles
            self.segment_radar_charts = radars
            # Persist clustered data for downstream pages
            self.clustered_data = self.df.to_dict("records")
            self.is_loading = False

    @rx.var
    def cluster_options(self) -> list[str]:
        """Returns a list of cluster names for the filter dropdown."""
        return ["All"] + list(self.cluster_mapping.values())

    @rx.var
    def avg_income(self) -> int:
        """Calculates the average annual income of all customers."""
        if self.df.empty:
            return 0
        return int(self.df["annual_income"].mean())

    @rx.var
    def avg_balance(self) -> int:
        """Calculates the average account balance of all customers."""
        if self.df.empty:
            return 0
        return int(self.df["account_balance"].mean())

    @rx.var
    def avg_credit_score(self) -> int:
        """Calculates the average credit score of all customers."""
        if self.df.empty:
            return 0
        return int(self.df["credit_score"].mean())

    @rx.var
    def largest_segment(self) -> str:
        """Identifies the largest customer segment."""
        if not self.segment_profiles:
            return "N/A"
        largest = max(self.segment_profiles, key=lambda p: p["size"])
        return largest["name"]

    @rx.var
    def highest_income_segment(self) -> str:
        """Identifies the segment with the highest average income."""
        if not self.segment_profiles:
            return "N/A"
        highest = max(
            self.segment_profiles, key=lambda p: p["features"]["annual_income"]
        )
        return highest["name"]


class DataCleaningState(rx.State):
    """Manages data upload, cleaning pipeline, and reporting."""

    is_processing: bool = False
    raw_csv_bytes: bytes | None = None
    raw_df: pd.DataFrame = pd.DataFrame()
    clean_df: pd.DataFrame = pd.DataFrame()
    report_df: pd.DataFrame = pd.DataFrame()

    total_rows: int = 0
    missing_values_count: int = 0
    outliers_count: int = 0
    corrections_count: int = 0

    step: str = "upload"  # upload -> control -> diagnose -> treat -> report

    @rx.event
    def reset_cleaning(self):
        self.is_processing = False
        self.raw_csv_bytes = None
        self.raw_df = pd.DataFrame()
        self.clean_df = pd.DataFrame()
        self.report_df = pd.DataFrame()
        self.total_rows = 0
        self.missing_values_count = 0
        self.outliers_count = 0
        self.corrections_count = 0
        self.step = "upload"

    @rx.event
    async def upload_csv(self, file: list[rx.UploadFile]):
        """Receive uploaded CSV list and store bytes backup (keep first)."""
        if not file:
            return
        uploaded = file[0]
        content = await uploaded.read()
        self.raw_csv_bytes = content
        from io import BytesIO

        try:
            self.raw_df = pd.read_csv(BytesIO(content))
        except Exception:
            # Try Excel just in case
            try:
                self.raw_df = pd.read_excel(BytesIO(content))
            except Exception:
                self.raw_df = pd.DataFrame()
        self.total_rows = int(len(self.raw_df))
        self.step = "control"

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        return df

    def _detect_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            df["__is_outlier__"] = False
            return df
        from scipy import stats

        z = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
        if isinstance(z, np.ndarray):
            mask = (z > 3).any(axis=1)
        else:
            mask = np.zeros(len(df), dtype=bool)
        df["__is_outlier__"] = mask
        return df

    def _diagnose_errors(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        # Missing values imputed
        for col in df_before.columns:
            if df_before[col].isnull().sum() > 0:
                records.append({
                    "type": "missing",
                    "column": col,
                    "action": "impute",
                    "strategy": "median_numeric_else_mode",
                })
        # Duplicates removed
        if len(df_before) > len(df_after):
            diff = int(len(df_before) - len(df_after))
            if diff > 0:
                records.append({
                    "type": "duplicate",
                    "column": "*",
                    "action": "drop_duplicates",
                    "count": diff,
                })
        return records

    @rx.event(background=True)
    async def run_cleaning(self):
        if self.raw_df.empty:
            return
        async with self:
            self.is_processing = True
            self.step = "diagnose"

        # Step 1-2: Control + Diagnose
        df = self._standardize_columns(self.raw_df)
        missing_before = int(df.isnull().sum().sum())
        df = self._detect_outliers_zscore(df)
        outliers_initial = int(df["__is_outlier__"].sum()) if "__is_outlier__" in df.columns else 0

        # Step 3: Treat
        df_treat = df.drop(columns=["__is_outlier__"]) if "__is_outlier__" in df.columns else df
        for col in df_treat.columns:
            if df_treat[col].dtype.kind in "biufc":
                if df_treat[col].isnull().any():
                    df_treat[col] = df_treat[col].fillna(df_treat[col].median())
            else:
                if df_treat[col].isnull().any():
                    df_treat[col] = df_treat[col].fillna(df_treat[col].mode().iloc[0])
        df_treat = df_treat.drop_duplicates()

        # Recompute outliers after imputation
        df_flag = self._detect_outliers_zscore(df_treat)
        outliers_after = int(df_flag["__is_outlier__"].sum()) if "__is_outlier__" in df_flag.columns else 0
        df_clean = df_treat.copy()

        # Step 4: Document
        records = self._diagnose_errors(self.raw_df, df_clean)
        # add outlier info
        if outliers_initial > 0:
            records.append({
                "type": "outlier",
                "column": "*",
                "action": "flagged",
                "z_threshold": 3,
                "count": outliers_initial,
                "remaining": outliers_after,
            })
        report_df = pd.DataFrame(records)

        async with self:
            self.clean_df = df_clean
            self.report_df = report_df
            self.missing_values_count = int(missing_before)
            self.outliers_count = int(outliers_initial)
            self.corrections_count = int(report_df.shape[0])
            self.step = "report"
            self.is_processing = False
        # Save cleaned CSV so other pages can load without cross-state checks
        try:
            os.makedirs("assets", exist_ok=True)
            df_clean.to_csv("assets/bank_customers_clean.csv", index=False)
        except Exception:
            pass

    @rx.event
    def download_clean_csv(self):
        if self.clean_df.empty:
            return
        return rx.download(
            data=self.clean_df.to_csv(index=False).encode("utf-8"),
            filename="bank_customers_clean.csv",
        )

    @rx.event
    def download_report_csv(self):
        if self.report_df.empty:
            return
        return rx.download(
            data=self.report_df.to_csv(index=False).encode("utf-8"),
            filename="data_cleaning_report.csv",
        )

    @rx.var
    def has_upload(self) -> bool:
        return self.raw_df is not None and not self.raw_df.empty

    @rx.var
    def has_clean(self) -> bool:
        return self.clean_df is not None and not self.clean_df.empty

    @rx.var
    def has_report(self) -> bool:
        return self.report_df is not None and not self.report_df.empty

    @rx.var
    def summary_stats(self) -> list[dict[str, str]]:
        return [
            {"name": "Total Rows", "value": str(self.total_rows)},
            {"name": "Missing Values", "value": str(self.missing_values_count)},
            {"name": "Outliers Detected", "value": str(self.outliers_count)},
            {"name": "Corrections Logged", "value": str(self.corrections_count)},
        ]